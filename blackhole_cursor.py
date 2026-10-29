"""
Black hole cursor v3   (Windows 10 2004+ for desktop warping; PyQt5 + PyOpenGL + numpy)

Mouse
    left click    shuttle           right click   planet (some have moons)
    middle click  star (some are binaries)
    side button 1 comet             side button 2 asteroid
    hold a button >0.35 s           gravity well: everything falls in faster
    Alt + drag                      flick: the drag direction picks which side the body comes in from,
                                    a longer drag gives it more spin (release to launch)

Hotkeys (Windows)
    Ctrl+Shift+Q  quit                      Ctrl+Shift+H  show / hide the real cursor
    Ctrl+Shift+M  cycle mode                Ctrl+Shift+T  cycle colour theme
    Ctrl+Shift+Up / Down  bigger / smaller hole

Settings live in  ~/.blackhole_cursor.json  (created on first run). The tray icon has the same controls.
"""
import sys
import time
import math
import json
import random
import signal
import atexit
from pathlib import Path
from collections import deque
import numpy as np
from PyQt5.QtWidgets import QApplication, QOpenGLWidget, QSystemTrayIcon, QMenu, QAction, QActionGroup
from PyQt5.QtCore import Qt, QTimer, QRect, QPointF
from PyQt5.QtGui import (QCursor, QSurfaceFormat, QImage, QPainter, QGuiApplication,
                         QIcon, QPixmap, QColor)
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

MAX_BODIES = 6
TRAIL_LEN = 8
SPAWN_R = 0.95     # bodies appear at the edge of the window
KILL_R = 0.26      # and are removed once they are well inside the shadow

SHUTTLE, PLANET, SUN, COMET, ASTEROID = 0, 1, 2, 3, 4
BODY_SCALE = {SHUTTLE: 0.12, PLANET: 0.085, SUN: 0.17, COMET: 0.10, ASTEROID: 0.09}   # starting size
BODY_FALL = {SHUTTLE: 0.13, PLANET: 0.10, SUN: 0.085, COMET: 0.16, ASTEROID: 0.12}    # infall speed
BODY_FLARE = {SHUTTLE: 0.25, PLANET: 0.45, SUN: 0.90, COMET: 0.30, ASTEROID: 0.15}    # meal -> flare
BODY_SPIKE = {SHUTTLE: 0.0, PLANET: 0.2, SUN: 1.0, COMET: 0.35, ASTEROID: 0.0}        # meal -> diffraction spikes
HEADING_KINDS = (SHUTTLE, COMET)                       # these point along their velocity

THEMES = {   # (cool, warm, hot)
    "classic": ((0.75, 0.10, 0.02), (1.00, 0.50, 0.12), (1.00, 0.92, 0.75)),
    "quasar":  ((0.15, 0.30, 0.85), (0.45, 0.70, 1.00), (0.92, 0.97, 1.00)),
    "toxic":   ((0.05, 0.45, 0.10), (0.35, 0.95, 0.20), (0.85, 1.00, 0.70)),
    "mono":    ((0.25, 0.25, 0.28), (0.65, 0.65, 0.70), (1.00, 1.00, 1.00)),
}
MODE_NAMES = ["Full", "Distortion only", "Minimal dark"]

# ----------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------
CONFIG_PATH = Path.home() / ".blackhole_cursor.json"
DEFAULT_CONFIG = {
    "size": 420,                 # overlay size in pixels (240..800)
    "fps": 60,                   # 15..120
    "low_power": False,          # 30 fps max, no bloom, desktop grabbed every other frame
    "theme": "classic",          # classic | quasar | toxic | mono
    "mode": 0,                   # 0 full, 1 distortion only, 2 minimal dark
    "mass": 1.0,                 # hole size multiplier (0.6..1.1)
    "intensity": 1.0,            # disk brightness
    "lens_strength": 1.0,        # how strongly the desktop is bent
    "disk_speed": 1.0,           # swirl speed of the accretion disk
    "idle_seconds": 5.0,         # grows and glows after this long without mouse movement
    "pause_on_fullscreen": True, # hide and give the real cursor back while a fullscreen app is active
    "hide_cursor": True,         # hide the real system cursor on start
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in DEFAULT_CONFIG:
                    if k in data and isinstance(data[k], type(DEFAULT_CONFIG[k])) or \
                            (k in data and isinstance(DEFAULT_CONFIG[k], float) and isinstance(data[k], int)):
                        cfg[k] = data[k]
        else:
            save_config(cfg)
    except Exception as e:
        print("Could not read config, using defaults:", e)
    cfg["size"] = int(_clamp(cfg["size"], 240, 800))
    cfg["fps"] = int(_clamp(cfg["fps"], 15, 120))
    cfg["mode"] = int(_clamp(cfg["mode"], 0, 2))
    cfg["mass"] = float(_clamp(cfg["mass"], 0.6, 1.1))
    cfg["intensity"] = float(_clamp(cfg["intensity"], 0.2, 3.0))
    cfg["lens_strength"] = float(_clamp(cfg["lens_strength"], 0.0, 3.0))
    cfg["disk_speed"] = float(_clamp(cfg["disk_speed"], 0.1, 4.0))
    cfg["idle_seconds"] = float(_clamp(cfg["idle_seconds"], 1.0, 600.0))
    if cfg["theme"] not in THEMES:
        cfg["theme"] = "classic"
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------
# shaders
# ----------------------------------------------------------------------------------------------
VERTEX_SRC = """
#version 120
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SRC = """
#version 120
uniform float time;
uniform vec2  resolution;
uniform sampler2D backdrop;   // screenshot of the desktop behind the window
uniform float lensOn;         // 1.0 when the backdrop is available
uniform float intro;          // seconds since launch
uniform float flare;          // jumps when the hole swallows something, then decays
uniform float wave;           // seconds since the last swallow (gravitational wave ring)
uniform float uSpike;         // diffraction spikes after a star is swallowed
uniform float holeScale;      // hole size multiplier (mass, idle growth, mode)
uniform float uMode;          // 0 full, 1 distortion only, 2 minimal dark
uniform float uQuality;       // 1 = bloom on, 0 = low power
uniform vec3  motion;         // xy = direction of mouse motion, z = amount 0..1
uniform vec3  cCool;          // colour theme
uniform vec3  cWarm;
uniform vec3  cHot;
uniform float uIntensity;     // disk brightness
uniform float uLensK;         // lens strength
uniform float uDiskSpeed;     // disk swirl speed
uniform vec4  bodies[6];      // x, y, heading-or-spin, scale  (scale <= 0 means unused)
uniform float kinds[6];       // floor = kind (0 shuttle, 1 planet, 2 sun, 3 comet, 4 asteroid), fract = seed
uniform vec4  trails[48];     // 8 samples per body: x, y, size, alpha

const float TILT   = 0.34;
const float R_HOLE = 0.30;
const float R_IN   = 0.32;
const float R_OUT  = 0.90;
const float PERIOD = 12.0;
const float SWIRL_K = 0.35;   // how strongly the desktop image twists around the hole

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float bump(float x, float center, float width) {
    float d = (x - center) / width;
    return exp(-d * d);
}

vec4 over(vec4 top, vec4 bottom) {
    return top + bottom * (1.0 - top.a);
}
vec4 emissive(vec3 color, float intensity) {
    return vec4(color * intensity, clamp(intensity, 0.0, 1.0));
}

vec2 rot(vec2 v, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

float swirl(vec2 dir, float r, float t) {
    float ang = -t * 0.3 / (max(r, 0.2) * sqrt(max(r, 0.2)));
    vec2 q = rot(dir, ang) * r;
    return vnoise(q * 10.0) * 0.55
         + vnoise(q * 22.0 + 7.3) * 0.30
         + vnoise(q * 45.0 + 1.7) * 0.15;
}

vec4 accretion(vec2 uv) {
    vec2 p = vec2(uv.x, uv.y / TILT);
    float r = length(p);
    vec2 dir = p / max(r, 1e-4);

    float f  = fract(time * uDiskSpeed / PERIOD);
    float tA = f * PERIOD;
    float tB = fract(f + 0.5) * PERIOD;
    float wA = 1.0 - abs(2.0 * f - 1.0);
    float n  = mix(swirl(dir, r, tB), swirl(dir, r, tA), wA);

    float bands = 0.8 + 0.2 * sin(r * 55.0 + n * 5.0);
    float heat  = clamp(1.0 - (r - R_IN) / (R_OUT - R_IN), 0.0, 1.0);

    float inner = smoothstep(R_IN, R_IN + 0.025, r);
    float outer = 1.0 - smoothstep(R_OUT - 0.35, R_OUT, r);

    float dop = 1.0 + 0.5 * dir.x;
    float inten = inner * outer
                * (0.25 + 1.6 * pow(heat, 1.6))
                * (0.45 + 0.9 * n) * bands
                * dop * dop;

    vec3 col = mix(cCool, cWarm, smoothstep(0.0, 0.55, heat));
    col      = mix(col, cHot,  smoothstep(0.55, 1.0, heat));
    // a meal heats the disk toward white
    col      = mix(col, cHot, clamp(flare * 0.5, 0.0, 0.6) * heat);

    return emissive(col, inten * (1.0 + 0.8 * flare) * uIntensity);
}

// The far side of the disk, bent over the top and under the bottom of the hole.
// Each pixel near the shadow is mapped back to a point on the far half of the disk plane.
vec4 lensedDisk(vec2 uv, float lr) {
    float a   = abs(uv.y) / max(lr, 1e-4);          // 1 straight above/below, 0 level with the hole
    float sg  = uv.x / max(lr, 1e-4);
    float top = step(0.0, uv.y);
    float k   = mix(3.2, 1.6, top);                 // the lower arc is thinner
    float rho = R_IN + (lr - R_HOLE) * k;
    vec4 d    = accretion(vec2(sg * rho, a * rho * TILT));
    float mask = smoothstep(R_HOLE + 0.002, R_HOLE + 0.03, lr)
               * (1.0 - smoothstep(0.40, 0.66, lr));
    float w    = smoothstep(0.12, 0.75, a) * mix(0.55, 1.0, top) * mask;
    return d * w;
}

// Relativistic jets blasting out of the poles, flickering, brighter after a meal.
vec4 jets(vec2 uv) {
    float ay    = abs(uv.y);
    float width = 0.015 + 0.05 * ay;
    float d     = uv.x / width;
    float core  = exp(-d * d);
    float len   = smoothstep(R_HOLE * 0.9, R_HOLE + 0.1, ay) * (1.0 - smoothstep(0.45, 0.95, ay));
    float pulse = 0.65 + 0.35 * vnoise(vec2(ay * 14.0 - time * 3.0, uv.x * 20.0));
    float i     = core * len * pulse * (0.10 + 0.55 * flare);
    return emissive(mix(cWarm, vec3(0.45, 0.65, 1.0), 0.6), i);
}

// Tiny particles drifting off the horizon.
float sparkle(vec2 uv) {
    float ang   = atan(uv.y, uv.x);
    float cells = 48.0;
    float cid   = floor((ang / 6.2831853 + 0.5) * cells);
    float acc   = 0.0;
    for (int k = 0; k < 3; k++) {
        float fk  = float(k);
        float h   = hash(vec2(cid, fk * 7.3));
        float ph  = fract(time * (0.12 + 0.10 * h) + h * 5.0);
        float rad = R_HOLE + 0.012 + ph * 0.28;
        float a0  = ((cid + 0.25 + 0.5 * hash(vec2(cid, fk * 3.1 + 1.0))) / cells - 0.5) * 6.2831853;
        vec2 pos  = rad * vec2(cos(a0), sin(a0));
        vec2 dd   = uv - pos;
        acc += exp(-dot(dd, dd) * 9000.0) * (1.0 - ph) * step(0.45, h);
    }
    return acc;
}

// ---------- gravitational lensing of the desktop ----------
vec2 toTex(vec2 s) {
    return vec2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
}

vec3 lensedBackdrop(vec2 uv, float lr) {
    float w   = 1.0 - smoothstep(0.40, 1.0, lr);
    float inv = uLensK * (1.0 + 0.4 * flare) * w / (lr * lr + 0.02);
    // frame-dragging twist: the closer to the hole, the more the image is rotated
    float ang = SWIRL_K * w * w / (lr * lr + 0.08);

    // gravitational wave: a ring that ripples outward after something is swallowed
    float lo  = length(uv);
    vec2 dirn = uv / max(lo, 1e-4);
    float amp = 0.03 * exp(-wave * 1.1) * bump(lo, wave * 0.9, 0.09)
              * (1.0 - smoothstep(0.60, 0.95, lo));

    vec2 suv = rot(uv, ang) + dirn * amp;

    // fast mouse movement smears the lensed image along the direction of travel
    vec3 c = vec3(0.0);
    for (int i = -1; i <= 1; i++) {
        vec2 b = suv + motion.xy * motion.z * 0.05 * float(i) * (0.3 + w);
        c.r += texture2D(backdrop, toTex(b * (1.0 - inv * 0.93))).r;
        c.g += texture2D(backdrop, toTex(b * (1.0 - inv))).g;
        c.b += texture2D(backdrop, toTex(b * (1.0 - inv * 1.07))).b;
    }
    c /= 3.0;
    c *= 0.35 + 0.65 * smoothstep(0.30, 0.75, lr);
    return c;
}

// ---------- falling bodies ----------
// Tidal spaghettification: returns the fragment position in a space that is
// stretched along the radial direction and squeezed sideways near the hole.
vec2 tidalWarp(vec2 uv, vec2 pos) {
    float r  = length(pos);
    vec2 rd  = pos / max(r, 1e-4);
    vec2 td  = vec2(-rd.y, rd.x);
    float tide    = 1.0 - smoothstep(0.32, 0.75, r);
    float stretch = 1.0 + 3.0 * tide;
    float squeeze = 1.0 + 1.3 * tide;
    vec2 d  = uv - pos;
    vec2 dd = vec2(dot(d, rd) / stretch, dot(d, td) * squeeze);
    return rd * dd.x + td * dd.y;
}

float pixelSize(float scale) {
    return 1.5 * (2.0 / min(resolution.x, resolution.y)) / scale;
}

// s = (x, y, heading, scale). Local space: nose points along +x, length ~2 units.
vec4 shuttleLayer(vec2 uv, vec4 s) {
    vec2 pos = s.xy;
    float r  = length(pos);
    vec2 dv  = tidalWarp(uv, pos);

    float c  = cos(s.z);
    float sn = sin(s.z);
    vec2 q = vec2(c * dv.x + sn * dv.y, -sn * dv.x + c * dv.y) / s.w;

    if (length(q) > 2.4) return vec4(0.0);

    float aa = pixelSize(s.w);
    float ay = abs(q.y);

    float dBody = (length(vec2((q.x + 0.1) / 0.92, q.y / 0.17)) - 1.0) * 0.17;
    float lead  = (ay - 0.1 - 0.5 * (0.1 - q.x)) * 0.894;
    float dWing = max(max(lead, -0.8 - q.x), q.x - 0.1);
    float dShip = min(dBody, dWing);
    float cov   = clamp(0.5 - dShip / aa, 0.0, 1.0);

    float t     = (-0.8 - q.x) / 0.7;
    float flick = 0.85 + 0.15 * sin(time * 40.0);
    float dFl   = max(max(ay - 0.11 * (1.0 - t) * flick, -t * 0.7), (t - 1.0) * 0.7);
    float flame = clamp(0.5 - dFl / aa, 0.0, 1.0) * (1.0 - cov);

    vec3 col = vec3(0.93, 0.94, 0.97);
    if (dWing < dBody) col = vec3(0.68, 0.70, 0.77);
    col = mix(col, vec3(0.07), smoothstep(0.55, 0.68, q.x));
    col = mix(col, vec3(0.10, 0.16, 0.32), 1.0 - smoothstep(0.05, 0.07, length(q - vec2(0.36, 0.0))));

    float near = 1.0 - smoothstep(0.30, 0.65, r);
    col = mix(col, col * vec3(1.0, 0.42, 0.18) + vec3(0.55, 0.12, 0.0), near);

    vec3 flameCol = mix(vec3(1.0, 0.5, 0.1), vec3(1.0, 0.95, 0.8), 1.0 - t);

    float a   = clamp(cov + flame, 0.0, 1.0);
    vec3  rgb = col * cov + flameCol * flame * 1.2;
    return vec4(min(rgb, vec3(a)), a);
}

// Three kinds of planet picked by seed: ocean world, ringed gas giant, red desert world.
// Ocean and desert worlds may have a moon that gets flung away as the system falls in.
vec4 planetLayer(vec2 uv, vec4 s, float seed) {
    vec2 pos = s.xy;
    float r  = length(pos);
    vec2 q   = tidalWarp(uv, pos) / s.w;
    float ql = length(q);
    if (ql > 5.2) return vec4(0.0);

    float aa  = pixelSize(s.w);
    float cov = clamp(0.5 - (ql - 1.0) / aa, 0.0, 1.0);

    vec3 n = vec3(q, sqrt(max(0.0, 1.0 - dot(q, q))));
    vec3 L = normalize(vec3(-0.55, 0.55, 0.62));
    float diff = clamp(dot(n, L), 0.0, 1.0);
    float lit  = 0.10 + 0.90 * diff;

    // longitude on the sphere, rotating with the spin phase
    float cl  = sqrt(max(1.0 - q.y * q.y, 1e-4));
    float lon = asin(clamp(q.x / cl, -1.0, 1.0)) + s.z;
    float ox  = seed * 40.0;

    float isEarth = 1.0 - step(0.34, seed);
    float isGas   = step(0.34, seed) * (1.0 - step(0.67, seed));
    float isMars  = step(0.67, seed);

    // ocean world
    float h = vnoise(vec2(lon * 2.5 + ox, q.y * 2.5)) * 0.65
            + vnoise(vec2(lon * 6.0 + ox, q.y * 6.0)) * 0.35;
    vec3 earth = mix(vec3(0.04, 0.18, 0.55), vec3(0.22, 0.52, 0.20), smoothstep(0.50, 0.56, h));
    earth = mix(earth, vec3(0.92, 0.95, 1.0), smoothstep(0.82, 0.92, abs(q.y)));
    float clouds = smoothstep(0.55, 0.75, vnoise(vec2(lon * 4.0 + ox + time * 0.2, q.y * 7.0)));
    earth = mix(earth, vec3(1.0), clouds * 0.55);

    // gas giant
    float band = sin(q.y * 9.0 + vnoise(vec2(lon * 2.0 + ox, q.y * 4.0)) * 2.5);
    vec3 gas = mix(vec3(0.86, 0.62, 0.36), vec3(0.58, 0.34, 0.20), band * 0.5 + 0.5);

    // desert world
    float m = vnoise(vec2(lon * 3.0 + ox, q.y * 3.0)) * 0.6
            + vnoise(vec2(lon * 9.0 + ox, q.y * 9.0)) * 0.4;
    vec3 mars = mix(vec3(0.72, 0.30, 0.14), vec3(0.42, 0.16, 0.09), m);

    vec3 surf = earth * isEarth + gas * isGas + mars * isMars;
    vec3 pc = surf * lit;
    pc += vec3(0.30, 0.55, 1.0) * pow(1.0 - n.z, 3.0) * 0.6 * diff * isEarth;   // atmosphere rim

    // heat and redshift near the hole
    float near = 1.0 - smoothstep(0.30, 0.65, r);
    pc = mix(pc, pc * vec3(1.0, 0.40, 0.15) + vec3(0.45, 0.08, 0.0), near);

    // rings (gas giant only), tilted ellipse; the far half hides behind the planet
    float cr = cos(0.35);
    float sr = sin(0.35);
    vec2 rq  = vec2(cr * q.x + sr * q.y, -sr * q.x + cr * q.y);
    rq.y /= 0.28;
    float rl = length(rq);
    float ringA = smoothstep(1.35, 1.42, rl) * (1.0 - smoothstep(1.98, 2.05, rl));
    ringA *= (0.55 + 0.45 * sin(rl * 38.0)) * 0.9 * isGas;
    float front = 1.0 - step(0.0, rq.y);
    ringA *= mix(1.0 - cov, 1.0, front);
    vec3 ringCol = vec3(0.88, 0.80, 0.62) * (0.45 + 0.55 * lit);
    ringCol = mix(ringCol, ringCol * vec3(1.0, 0.40, 0.15) + vec3(0.45, 0.08, 0.0), near);

    vec4 P = vec4(pc * cov, cov);
    vec4 R = vec4(ringCol * ringA, ringA);
    vec4 res = mix(over(P, R), over(R, P), front);
    vec4 base = vec4(min(res.rgb, vec3(res.a)), res.a);

    // moon: orbits the planet, is flung outwards and fades as the system falls in
    float hasMoon = step(0.5, fract(seed * 7.0)) * (1.0 - isGas);
    float ma   = s.z * 1.9 + seed * 6.0;
    float tear = 1.0 - smoothstep(0.45, 0.80, r);
    vec2  mc   = vec2(cos(ma), sin(ma) * 0.35) * 1.75 * (1.0 + 1.6 * tear);
    float mr   = 0.22;
    vec2  mq   = (q - mc) / mr;
    float mql  = length(mq);
    float mcov = clamp(0.5 - (mql - 1.0) / (aa / mr), 0.0, 1.0);
    float mnz  = sqrt(max(0.0, 1.0 - dot(mq, mq)));
    float mdif = clamp(dot(vec3(mq, mnz), L), 0.0, 1.0);
    vec3  mcol = vec3(0.64, 0.62, 0.60) * (0.10 + 0.90 * mdif) * (0.8 + 0.4 * vnoise(mq * 4.0 + ox));
    mcol = mix(mcol, mcol * vec3(1.0, 0.40, 0.15) + vec3(0.45, 0.08, 0.0), near);
    float mA = mcov * hasMoon * smoothstep(0.42, 0.70, r);
    vec4 M = vec4(mcol * mA, mA);
    float mfront = step(sin(ma), 0.0);
    vec4 outc = mix(over(base, M), over(M, base), mfront);
    return vec4(min(outc.rgb, vec3(outc.a)), outc.a);
}

// One star with limb darkening, boiling granulation and a flickering corona.
// q is in star-local units (disc radius 1); aa is the pixel size in those units.
vec4 starDisc(vec2 q, float seed, float r, float aa) {
    float ql   = length(q);
    float disc = clamp(0.5 - (ql - 1.0) / aa, 0.0, 1.0);

    vec3 hotC  = vec3(1.0, 0.95, 0.70);
    vec3 coolC = vec3(1.0, 0.50, 0.10);
    if (seed > 0.85) {            // blue-white star
        hotC  = vec3(0.88, 0.94, 1.0);
        coolC = vec3(0.35, 0.55, 1.0);
    } else if (seed > 0.60) {     // red giant
        hotC  = vec3(1.0, 0.70, 0.40);
        coolC = vec3(0.90, 0.20, 0.05);
    }

    float z    = sqrt(max(0.0, 1.0 - ql * ql));
    float limb = 0.55 + 0.45 * z;
    float g    = vnoise(q * 7.0 + vec2(time * 0.3, seed * 30.0)) * 0.6
               + vnoise(q * 16.0 - vec2(time * 0.5, 0.0)) * 0.4;
    vec3 col   = mix(coolC, hotC, clamp(limb * (0.65 + 0.55 * g), 0.0, 1.0));

    float rays = 0.85 + 0.15 * sin(atan(q.y, q.x + 1e-5) * 9.0 + time * 1.5);
    float cor  = exp(-max(ql - 1.0, 0.0) * 3.2) * 0.7 * rays;

    // redshift and dimming as it falls in
    float near = 1.0 - smoothstep(0.30, 0.65, r);
    col  = mix(col, col * vec3(1.0, 0.35, 0.10), near);
    vec3 corCol = mix(coolC, coolC * vec3(1.0, 0.35, 0.10), near);

    float a   = clamp(max(disc, cor), 0.0, 1.0);
    vec3 rgb  = col * disc + corCol * cor * (1.0 - disc);
    return vec4(min(rgb, vec3(a)), a);
}

// A star, or (for some seeds) a binary pair orbiting each other as they fall in.
vec4 sunLayer(vec2 uv, vec4 s, float seed) {
    vec2 pos = s.xy;
    float r  = length(pos);
    vec2 q   = tidalWarp(uv, pos) / s.w;
    if (length(q) > 3.6) return vec4(0.0);

    float aa  = pixelSize(s.w);
    float bin = step(0.6, fract(seed * 11.0));
    if (bin < 0.5) return starDisc(q, seed, r, aa);

    float ang = s.z * 2.0;
    vec2  d   = vec2(cos(ang), sin(ang));
    float s1  = 0.70;
    float s2  = 0.52;
    vec4 A = starDisc((q - 0.95 * d) / s1, seed, r, aa / s1);
    vec4 B = starDisc((q + 0.95 * d) / s2, fract(seed * 3.7 + 0.31), r, aa / s2);
    float front = step(0.0, sin(ang));
    return mix(over(A, B), over(B, A), front);
}

// Comet: bright nucleus and coma, a thin blue ion tail and a curved yellowish dust tail.
vec4 cometLayer(vec2 uv, vec4 s, float seed) {
    vec2 pos = s.xy;
    float r  = length(pos);
    vec2 dv  = tidalWarp(uv, pos);
    float c  = cos(s.z);
    float sn = sin(s.z);
    vec2 q = vec2(c * dv.x + sn * dv.y, -sn * dv.x + c * dv.y) / s.w;
    if (q.x < -7.0 || q.x > 2.5 || abs(q.y) > 2.5) return vec4(0.0);

    float aa  = pixelSize(s.w);
    float nuc = length(q);
    float cov = clamp(0.5 - (nuc - 0.45) / aa, 0.0, 1.0);
    float coma = exp(-nuc * nuc * 1.6) * 0.9;

    float t     = max(-q.x, 0.0);
    float start = smoothstep(-0.4, 0.2, -q.x);
    float wIon  = 0.12 + 0.03 * t;
    float ion   = exp(-(q.y / wIon) * (q.y / wIon)) * exp(-t * 0.45) * start;
    ion *= 0.85 + 0.15 * vnoise(vec2(q.x * 6.0 + time * 4.0, q.y * 8.0));
    float yd    = q.y + 0.05 * t * t;
    float wDust = 0.22 + 0.10 * t;
    float dust  = exp(-(yd / wDust) * (yd / wDust)) * exp(-t * 0.55) * start * 0.55;

    vec3 rgb = vec3(0.45, 0.80, 1.0) * ion * 0.9
             + vec3(1.0, 0.90, 0.70) * dust * 0.8
             + vec3(1.0, 0.97, 0.90) * coma * 0.9;
    rgb = rgb * (1.0 - cov) + vec3(0.92, 0.94, 1.0) * cov;
    float a = clamp(max(cov, clamp(coma + ion * 0.9 + dust * 0.8, 0.0, 1.0)), 0.0, 1.0);

    float near = 1.0 - smoothstep(0.30, 0.65, r);
    rgb = mix(rgb, rgb * vec3(1.0, 0.40, 0.15) + vec3(0.40, 0.07, 0.0) * a, near);
    return vec4(min(rgb, vec3(a)), a);
}

// Asteroid: lumpy tumbling rock with fake craters.
vec4 asteroidLayer(vec2 uv, vec4 s, float seed) {
    vec2 pos = s.xy;
    float r  = length(pos);
    vec2 q   = rot(tidalWarp(uv, pos), -s.z) / s.w;
    float ql = length(q);
    if (ql > 2.2) return vec4(0.0);

    vec2 dir = q / max(ql, 1e-4);
    float ox = seed * 37.0;
    float rad = 0.78 + 0.45 * vnoise(dir * 1.6 + ox) + 0.12 * vnoise(dir * 4.5 + ox * 1.7);
    float aa  = pixelSize(s.w);
    float cov = clamp(0.5 - (ql - rad) / aa, 0.0, 1.0);

    vec2 qn   = q / rad;
    float nz  = sqrt(max(0.0, 1.0 - dot(qn, qn)));
    float cra = vnoise(q * 5.0 + ox) - 0.5;
    vec3 n    = normalize(vec3(qn + cra * 0.35, nz + 0.001));
    vec3 L    = normalize(vec3(-0.55, 0.55, 0.62));
    float diff = clamp(dot(n, L), 0.0, 1.0);
    vec3 base = mix(vec3(0.33, 0.30, 0.28), vec3(0.55, 0.50, 0.45), vnoise(q * 3.0 + ox));
    vec3 pc   = base * (0.08 + 0.92 * diff);

    float near = 1.0 - smoothstep(0.30, 0.65, r);
    pc = mix(pc, pc * vec3(1.0, 0.40, 0.15) + vec3(0.45, 0.08, 0.0), near);
    return vec4(pc * cov, cov);
}

vec4 bodyLayer(vec2 uv, vec4 s, float kv, float lr) {
    if (s.w <= 0.0) return vec4(0.0);
    float kind = floor(kv);
    float seed = fract(kv);
    float r    = length(s.xy);

    vec4 res;
    if (kind < 0.5)      res = shuttleLayer(uv, s);
    else if (kind < 1.5) res = planetLayer(uv, s, seed);
    else if (kind < 2.5) res = sunLayer(uv, s, seed);
    else if (kind < 3.5) res = cometLayer(uv, s, seed);
    else                 res = asteroidLayer(uv, s, seed);

    float fade = (1.0 - smoothstep(0.78, 0.93, r))        // fade in at the spawn edge
               * (1.0 - smoothstep(0.80, 0.98, lr))       // never clipped by the window edge
               * smoothstep(0.26, 0.38, r);               // dissolve into the shadow
    return res * fade;
}

vec3 trailColor(float k) {
    if (k < 0.5) return vec3(0.65, 0.80, 1.00);
    if (k < 1.5) return vec3(0.80, 0.65, 0.45);
    if (k < 2.5) return vec3(1.00, 0.60, 0.20);
    if (k < 3.5) return vec3(0.50, 0.80, 1.00);
    return vec3(0.60, 0.55, 0.50);
}

// ---------- the cursor that gets swallowed at launch ----------
float edgeD(vec2 p, vec2 a, vec2 b) {
    vec2 ba = b - a;
    return -(ba.x * (p.y - a.y) - ba.y * (p.x - a.x)) / length(ba);
}
float segD(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

vec4 cursorLayer(vec2 uv, float t) {
    float e = clamp((t - 0.5) / 1.9, 0.0, 1.0);
    e = e * e * (3.0 - 2.0 * e);

    float sz0 = 0.13;
    vec2  c0  = vec2(0.25, -0.5) * sz0;           // arrow centre, tip sits at the hole centre
    float th  = atan(c0.y, c0.x) + 12.0 * e * e;  // spirals inward
    float rr  = length(c0) * (1.0 - 0.9 * e);
    vec2 pos  = rr * vec2(cos(th), sin(th));
    float sz  = sz0 * (1.0 - 0.7 * e);

    // tidal stretching along the radial direction
    float rl = max(length(pos), 1e-4);
    vec2 rd  = pos / rl;
    vec2 td  = vec2(-rd.y, rd.x);
    vec2 d   = uv - pos;
    vec2 dd  = vec2(dot(d, rd) / (1.0 + 5.0 * e), dot(d, td) * (1.0 + 1.5 * e));
    vec2 w   = rd * dd.x + td * dd.y;

    vec2 q = rot(w, -4.0 * e * e * 3.14159) / sz + vec2(0.25, -0.5);

    vec2 A = vec2(0.0, 0.0);
    vec2 B = vec2(0.0, -1.0);
    vec2 C = vec2(0.7, -0.7);
    float dHead = max(edgeD(q, A, B), max(edgeD(q, B, C), edgeD(q, C, A)));
    float dTail = segD(q, vec2(0.30, -0.72), vec2(0.50, -1.12)) - 0.09;
    float ds = min(dHead, dTail);

    float aa   = 1.5 * (2.0 / min(resolution.x, resolution.y)) / sz;
    float covI = clamp(0.5 - ds / aa, 0.0, 1.0);
    float covO = clamp(0.5 - (ds - 0.07) / aa, 0.0, 1.0);
    vec3 col = mix(vec3(0.02), vec3(1.0), covI);
    col = mix(col, col * vec3(1.0, 0.4, 0.15) + vec3(0.45, 0.08, 0.0), e);   // redshift
    return vec4(col * covO, covO);
}

void main() {
    vec2 uvO = (gl_FragCoord.xy * 2.0 - resolution) / min(resolution.x, resolution.y);
    float lrO = length(uvO);

    // the black hole grows out of the cursor during the first ~2.4 s
    float g  = max(smoothstep(0.0, 2.4, intro), 0.002) * holeScale;
    vec2 uv  = uvO / g;
    float lr = length(uv);

    // modes: 0 full, 1 distortion only, 2 minimal dark
    float dk = (uMode < 0.5) ? 1.0 : ((uMode < 1.5) ? 0.0 : 0.3);
    float fx = (uMode < 0.5) ? 1.0 : 0.0;
    float ps = (uMode < 0.5) ? 1.0 : ((uMode < 1.5) ? 0.4 : 0.7);

    vec4 disk = accretion(uv);
    vec4 arc  = lensedDisk(uv, lr);
    float backMask = smoothstep(-0.012, 0.012, uv.y);
    float hole = 1.0 - smoothstep(R_HOLE - 0.006, R_HOLE + 0.004, lr);

    // bloom: a few blurred taps of the disk
    vec4 bloom = vec4(0.0);
    if (uQuality > 0.5 && dk > 0.0) {
        bloom = (accretion(uv + vec2(0.07, 0.0)) + accretion(uv - vec2(0.07, 0.0))
               + accretion(uv + vec2(0.0, 0.05)) + accretion(uv - vec2(0.0, 0.05))) * 0.07;
    }

    float up  = clamp(uv.y / max(lr, 1e-4) * 0.5 + 0.5, 0.0, 1.0);
    float dop = 1.0 + 0.5 * uv.x / max(lr, 1e-4);
    vec4 lens = emissive(mix(cWarm, cHot, 0.3),
                         bump(lr, R_HOLE + 0.06, 0.05) * (0.25 + 0.95 * up) * dop * dop
                         * 0.5 * (1.0 + 0.5 * flare));
    vec4 halo = emissive(cCool * 1.2, bump(lr, 0.42, 0.22) * (0.16 + 0.10 * flare));

    vec4 scene = (halo + lens + disk * backMask + arc + bloom) * dk + jets(uv) * fx;
    scene.rgb *= (1.0 - hole);
    scene.a    = clamp(scene.a, 0.0, 1.0) * (1.0 - hole) + hole;

    vec4 col = over(disk * (1.0 - backMask) * dk, scene);

    // photon ring plus a thinner, dimmer second image of it
    vec4 photon  = emissive(cHot, bump(lr, R_HOLE + 0.012, 0.011) * 1.4 * ps);
    vec4 photon2 = emissive(mix(cHot, cWarm, 0.4), bump(lr, R_HOLE + 0.036, 0.005) * 0.45 * ps);
    col.rgb += photon.rgb + photon2.rgb;
    col.a    = max(col.a, max(photon.a, photon2.a));

    col *= 1.0 - smoothstep(0.80, 0.98, lr);
    col = clamp(col, 0.0, 1.0);
    col.a = max(col.a, max(col.r, max(col.g, col.b)));

    float bgA = (1.0 - smoothstep(0.82, 0.99, lrO)) * lensOn;
    vec4 bg = vec4(lensedBackdrop(uvO, lr) * bgA, bgA);
    vec4 result = col + bg * (1.0 - col.a);

    // motion trails behind falling bodies
    if (fx > 0.5) {
        float edge = 1.0 - smoothstep(0.80, 0.98, lr);
        for (int i = 0; i < 6; i++) {
            if (bodies[i].w > 0.0) {
                vec3 tc = trailColor(floor(kinds[i]));
                for (int j = 0; j < 8; j++) {
                    vec4 t = trails[i * 8 + j];
                    if (t.w > 0.0) {
                        float tr   = length(t.xy);
                        float fade = smoothstep(0.28, 0.42, tr) * (1.0 - smoothstep(0.78, 0.93, tr));
                        vec2  dt   = uv - t.xy;
                        float a    = exp(-dot(dt, dt) / (t.z * t.z)) * t.w * fade;
                        result.rgb += tc * a * (1.0 - hole) * edge;
                    }
                }
            }
        }
    }

    // falling bodies, hidden by the shadow once they cross the horizon
    for (int i = 0; i < 6; i++) {
        vec4 b = bodyLayer(uv, bodies[i], kinds[i], lr) * (1.0 - hole);
        result = b + result * (1.0 - b.a);
    }

    // hawking-style sparkle drifting off the horizon
    if (fx > 0.5) {
        float sp = sparkle(uv) * (1.0 - hole) * (1.0 - smoothstep(0.55, 0.9, lr));
        result.rgb += mix(cHot, vec3(0.6, 0.8, 1.0), 0.5) * sp * 0.9;
    }

    // diffraction spikes when a star is swallowed
    if (fx > 0.5 && uSpike > 0.002) {
        vec2 sd = rot(uv, 0.7853982);
        float s1 = exp(-abs(uv.y) * 90.0) * exp(-abs(uv.x) * 4.0);
        float s2 = exp(-abs(uv.x) * 90.0) * exp(-abs(uv.y) * 4.0);
        float s3 = exp(-abs(sd.y) * 110.0) * exp(-abs(sd.x) * 6.0);
        float s4 = exp(-abs(sd.x) * 110.0) * exp(-abs(sd.y) * 6.0);
        float spk = (s1 + s2 + 0.5 * (s3 + s4)) * uSpike
                  * (1.0 - smoothstep(0.55, 0.95, lr)) * (1.0 - smoothstep(0.82, 0.99, lrO));
        result.rgb += cHot * spk;
    }

    // the real cursor, being swallowed at launch
    if (intro < 3.0) {
        vec4 c = cursorLayer(uvO, intro) * (1.0 - hole);
        result = c + result * (1.0 - c.a);
    }

    result = clamp(result, 0.0, 1.0);
    result.a = max(result.a, max(result.r, max(result.g, result.b)));
    gl_FragColor = result;
}
"""

UNIFORM_NAMES = [
    "time", "resolution", "backdrop", "lensOn", "intro", "flare", "wave", "uSpike", "holeScale",
    "uMode", "uQuality", "motion", "cCool", "cWarm", "cHot", "uIntensity", "uLensK", "uDiskSpeed",
    "bodies", "kinds", "trails",
]


# ----------------------------------------------------------------------------------------------
# simulation objects
# ----------------------------------------------------------------------------------------------
class Body:
    """One object on an inward spiral (angular momentum is conserved)."""

    def __init__(self, kind, fling=None):
        self.kind = kind
        self.base_scale = BODY_SCALE[kind]
        self.fall = BODY_FALL[kind]
        self.r = SPAWN_R
        self.theta = random.uniform(0.0, 2.0 * math.pi)
        omega0 = random.uniform(0.9, 1.5) * random.choice((-1.0, 1.0))
        if fling is not None:
            # (angle, strength 0..1): the drag picks the side it comes in from, longer drag = more spin
            self.theta = fling[0]
            omega0 = (0.5 + 1.5 * fling[1]) * random.choice((-1.0, 1.0))
        if kind in (PLANET, SUN, ASTEROID):
            omega0 *= 0.7
        elif kind == COMET:
            omega0 *= 0.45
        self.L = SPAWN_R * SPAWN_R * omega0
        self.seed = random.uniform(0.0, 0.99)
        self.spin = random.uniform(0.0, 2.0 * math.pi)
        self.spin_rate = random.uniform(0.5, 1.2) * random.choice((-1.0, 1.0))
        self.heading = 0.0
        self.scale = self.base_scale
        self.trail = deque(maxlen=TRAIL_LEN)     # newest first: (x, y, scale)
        self._tt = 0.0
        self._refresh(0.0, omega0)

    def _refresh(self, rdot, thdot):
        k = min(1.0, max(0.3, 0.35 + 0.65 * (self.r - 0.30) / 0.65))
        self.scale = self.base_scale * k
        if self.kind in HEADING_KINDS:   # shuttles and comets point along their velocity
            c, s = math.cos(self.theta), math.sin(self.theta)
            vx = rdot * c - self.r * thdot * s
            vy = rdot * s + self.r * thdot * c
            self.heading = math.atan2(vy, vx)

    def step(self, dt, pull=1.0):
        dt = dt * pull
        rdot = -self.fall * (SPAWN_R / self.r) ** 2      # falls faster and faster
        self.r += rdot * dt
        thdot = self.L / (self.r * self.r)               # spins faster and faster
        self.theta += thdot * dt
        self.spin += self.spin_rate * dt * (SPAWN_R / self.r)
        self._refresh(rdot, thdot)
        self._tt += dt
        if self._tt >= 0.03:
            self._tt = 0.0
            self.trail.appendleft((self.r * math.cos(self.theta), self.r * math.sin(self.theta), self.scale))
        return self.r > KILL_R

    def packed(self):
        z = self.heading if self.kind in HEADING_KINDS else self.spin
        return (self.r * math.cos(self.theta), self.r * math.sin(self.theta), z, self.scale)

    def kind_value(self):
        return self.kind + self.seed


BUTTONS_WIN = ((SHUTTLE, 0x01), (PLANET, 0x02), (SUN, 0x04), (COMET, 0x05), (ASTEROID, 0x06))


class ClickWatcher:
    """Global click detection (the overlay itself is click-through).
    poll() returns a list of (kind, pressed) events.
    left -> shuttle, right -> planet, middle -> sun, side1 -> comet, side2 -> asteroid."""

    def __init__(self):
        self.mode = None
        self.down = {k: False for k, _ in BUTTONS_WIN}
        self._events = deque()
        try:
            import ctypes
            self._gaks = ctypes.windll.user32.GetAsyncKeyState
            self.mode = "win"
        except Exception:
            try:
                from pynput import mouse
                self._mouse = mouse
                self._listener = mouse.Listener(on_click=self._on_click)
                self._listener.daemon = True
                self._listener.start()
                self.mode = "pynput"
            except Exception:
                print("No global click detection available "
                      "(on non-Windows: pip install pynput).")

    def _on_click(self, x, y, button, pressed):
        B = self._mouse.Button
        mapping = ((getattr(B, "left", None), SHUTTLE), (getattr(B, "right", None), PLANET),
                   (getattr(B, "middle", None), SUN), (getattr(B, "x1", None), COMET),
                   (getattr(B, "x2", None), ASTEROID))
        for b, kind in mapping:
            if b is not None and button == b:
                self._events.append((kind, bool(pressed)))
                return

    def poll(self):
        out = []
        if self.mode == "win":
            for kind, vk in BUTTONS_WIN:
                down = bool(self._gaks(vk) & 0x8000)
                if down != self.down[kind]:
                    out.append((kind, down))
                    self.down[kind] = down
        elif self.mode == "pynput":
            while self._events:
                kind, pressed = self._events.popleft()
                self.down[kind] = pressed
                out.append((kind, pressed))
        return out

    def key_down(self, vk):
        """True while a virtual key is held (Windows only)."""
        if self.mode == "win":
            return bool(self._gaks(vk) & 0x8000)
        return False


# ----------------------------------------------------------------------------------------------
# Windows helpers
# ----------------------------------------------------------------------------------------------
_cursor_hidden = False
_atexit_registered = False


def hide_system_cursor():
    """Replace every system cursor with a blank one (Windows)."""
    global _cursor_hidden, _atexit_registered
    try:
        import ctypes
        u = ctypes.windll.user32
        u.CreateCursor.restype = ctypes.c_void_p
        u.CreateCursor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p]
        u.SetSystemCursor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        for cid in (32512, 32513, 32514, 32515, 32516, 32642, 32643, 32644,
                    32645, 32646, 32648, 32649, 32650, 32651):
            and_mask = (ctypes.c_ubyte * 128)(*([0xFF] * 128))
            xor_mask = (ctypes.c_ubyte * 128)()
            h = u.CreateCursor(None, 0, 0, 32, 32, and_mask, xor_mask)
            if h:
                u.SetSystemCursor(h, cid)
        _cursor_hidden = True
        if not _atexit_registered:
            atexit.register(restore_system_cursor)
            _atexit_registered = True
    except Exception:
        pass


def restore_system_cursor():
    global _cursor_hidden
    try:
        import ctypes
        ctypes.windll.user32.SystemParametersInfoW(0x0057, 0, None, 0)  # SPI_SETCURSORS
        _cursor_hidden = False
    except Exception:
        pass


def foreground_is_fullscreen():
    """True when the active window covers its whole monitor (games, fullscreen video). Windows only."""
    import ctypes
    from ctypes import wintypes

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = ctypes.c_void_p
    u.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    u.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    u.MonitorFromWindow.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    u.MonitorFromWindow.restype = ctypes.c_void_p
    u.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]

    hwnd = u.GetForegroundWindow()
    if not hwnd:
        return False
    buf = ctypes.create_unicode_buffer(64)
    u.GetClassNameW(hwnd, buf, 64)
    if buf.value in ("Progman", "WorkerW", "Shell_TrayWnd"):
        return False
    rect = wintypes.RECT()
    if not u.GetWindowRect(hwnd, ctypes.byref(rect)):
        return False
    hmon = u.MonitorFromWindow(hwnd, 2)          # MONITOR_DEFAULTTONEAREST
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    if not u.GetMonitorInfoW(hmon, ctypes.byref(mi)):
        return False
    m = mi.rcMonitor
    return (rect.left <= m.left and rect.top <= m.top
            and rect.right >= m.right and rect.bottom >= m.bottom)


# ----------------------------------------------------------------------------------------------
# the overlay
# ----------------------------------------------------------------------------------------------
class BlackHoleCursor(QOpenGLWidget):
    SIZE = 420

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.SIZE = cfg["size"]
        self.mode = cfg["mode"]
        self.theme = cfg["theme"]
        self.on_change = None            # set by the tray to keep its check marks in sync

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.resize(self.SIZE, self.SIZE)

        self._t0 = time.perf_counter()
        self._last = self._t0
        self._last_pos = None
        self._frame = None
        self._frame_n = 0
        self._tex_size = (1, 1)
        self.lens_ok = False
        self.bodies = []
        self.flare = 0.0
        self.wave = 99.0
        self.spike = 0.0
        self.pull = 0.0
        self.hole_scale = 1.0
        self.paused = False
        self._cursor_was_hidden = False
        self._fs_t = 0.0
        self._hk = {}
        self._hold = {}                  # kind -> time the button went down
        self._flick = {}                 # kind -> cursor position at press (Alt+drag)
        self._last_cursor = None
        self._last_move = self._t0
        self._vel = (0.0, 0.0)
        self._mdir = (1.0, 0.0)
        self._mamt = 0.0
        self.clicks = ClickWatcher()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self._apply_fps()
        self.timer.start()

    # ---------------- settings actions ----------------
    def _apply_fps(self):
        fps = self.cfg["fps"]
        if self.cfg["low_power"]:
            fps = min(fps, 30)
        self.timer.setInterval(max(1, int(1000 / fps)))

    def _changed(self):
        save_config(self.cfg)
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass

    def set_mode(self, i):
        self.mode = int(i) % 3
        self.cfg["mode"] = self.mode
        if self.mode != 0:
            self.bodies.clear()
        self._changed()

    def cycle_mode(self):
        self.set_mode(self.mode + 1)

    def set_theme(self, name):
        if name in THEMES:
            self.theme = name
            self.cfg["theme"] = name
            self._changed()

    def cycle_theme(self):
        names = list(THEMES)
        self.set_theme(names[(names.index(self.theme) + 1) % len(names)])

    def toggle_cursor(self):
        if _cursor_hidden:
            restore_system_cursor()
            self.cfg["hide_cursor"] = False
        elif self.lens_ok:
            hide_system_cursor()
            self.cfg["hide_cursor"] = True
        self._changed()

    def toggle_low_power(self):
        self.cfg["low_power"] = not self.cfg["low_power"]
        self._apply_fps()
        self._changed()

    def change_mass(self, delta):
        self.cfg["mass"] = float(_clamp(self.cfg["mass"] + delta, 0.6, 1.1))
        self._changed()

    # ---------------- capture ----------------
    def exclude_from_capture(self):
        try:
            import ctypes
            ok = ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11)
            self.lens_ok = bool(ok)
        except Exception:
            self.lens_ok = False
        if not self.lens_ok:
            print("Desktop warping needs Windows 10 (2004+); running without it.")

    def grab_backdrop(self, gx, gy):
        S = self.SIZE
        center = QRect(gx, gy, S, S).center()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        geo = screen.geometry()
        rect = QRect(gx, gy, S, S)
        inter = rect.intersected(geo)
        if inter.isEmpty():
            return None
        pm = screen.grabWindow(0, inter.x() - geo.x(), inter.y() - geo.y(),
                               inter.width(), inter.height())
        if inter == rect:
            return pm.toImage().convertToFormat(QImage.Format_RGB32)
        dpr = screen.devicePixelRatio()
        canvas = QImage(int(S * dpr), int(S * dpr), QImage.Format_RGB32)
        canvas.setDevicePixelRatio(dpr)
        canvas.fill(Qt.black)
        p = QPainter(canvas)
        p.drawPixmap(QPointF(inter.x() - gx, inter.y() - gy), pm)
        p.end()
        return canvas

    # ---------------- GL ----------------
    def initializeGL(self):
        self.program = compileProgram(
            compileShader(VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self.a_pos = glGetAttribLocation(self.program, "position")
        self.U = {n: glGetUniformLocation(self.program, n) for n in UNIFORM_NAMES}

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     b"\x00\x00\x00\xff")
        self._tex_size = (1, 1)

        glDisable(GL_BLEND)

    def _upload_frame(self):
        img = self._frame
        self._frame = None
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        data = ptr.asstring(img.byteCount())
        size = (img.width(), img.height())
        if size == self._tex_size:
            # same size as last frame: update in place, much cheaper than reallocating
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size[0], size[1],
                            GL_BGRA, GL_UNSIGNED_BYTE, data)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
                         GL_BGRA, GL_UNSIGNED_BYTE, data)
            self._tex_size = size

    def paintGL(self):
        U = self.U
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        if self._frame is not None:
            self._upload_frame()
        glUniform1i(U["backdrop"], 0)
        glUniform1f(U["lensOn"], 1.0 if self.lens_ok else 0.0)

        now = time.perf_counter()
        glUniform1f(U["intro"], min(now - self._t0, 10.0))
        glUniform1f(U["time"], (now - self._t0) % 120.0)
        glUniform1f(U["flare"], self.flare)
        glUniform1f(U["wave"], min(self.wave, 99.0))
        glUniform1f(U["uSpike"], self.spike)
        glUniform1f(U["holeScale"], self.hole_scale)
        glUniform1f(U["uMode"], float(self.mode))
        glUniform1f(U["uQuality"], 0.0 if self.cfg["low_power"] else 1.0)
        glUniform3f(U["motion"], self._mdir[0], self._mdir[1], self._mamt)

        cool, warm, hot = THEMES[self.theme]
        glUniform3f(U["cCool"], *cool)
        glUniform3f(U["cWarm"], *warm)
        glUniform3f(U["cHot"], *hot)
        glUniform1f(U["uIntensity"], self.cfg["intensity"])
        glUniform1f(U["uLensK"], 0.17 * self.cfg["lens_strength"])
        glUniform1f(U["uDiskSpeed"], self.cfg["disk_speed"])

        arr = np.zeros((MAX_BODIES, 4), dtype=np.float32)
        kinds = np.zeros(MAX_BODIES, dtype=np.float32)
        trails = np.zeros((MAX_BODIES * TRAIL_LEN, 4), dtype=np.float32)
        for i, b in enumerate(self.bodies[:MAX_BODIES]):
            arr[i] = b.packed()
            kinds[i] = b.kind_value()
            for j, (x, y, sc) in enumerate(b.trail):
                trails[i * TRAIL_LEN + j] = (x, y, max(sc * 0.55 * (1.0 - 0.35 * j / TRAIL_LEN), 0.01),
                                             0.55 * (1.0 - j / float(TRAIL_LEN)))
        glUniform4fv(U["bodies"], MAX_BODIES, arr)
        glUniform1fv(U["kinds"], MAX_BODIES, kinds)
        glUniform4fv(U["trails"], MAX_BODIES * TRAIL_LEN, trails)

        dpr = self.devicePixelRatioF()
        glUniform2f(U["resolution"], self.width() * dpr, self.height() * dpr)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableVertexAttribArray(self.a_pos)
        glVertexAttribPointer(self.a_pos, 2, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableVertexAttribArray(self.a_pos)

    # ---------------- input ----------------
    def _edge(self, name, cond):
        prev = self._hk.get(name, False)
        self._hk[name] = cond
        return cond and not prev

    def poll_hotkeys(self):
        c = self.clicks
        if c.mode != "win":
            return
        mod = c.key_down(0x11) and c.key_down(0x10)          # Ctrl + Shift
        if self._edge("q", mod and c.key_down(0x51)):
            QApplication.quit()
        if self._edge("h", mod and c.key_down(0x48)):
            self.toggle_cursor()
        if self._edge("m", mod and c.key_down(0x4D)):
            self.cycle_mode()
        if self._edge("t", mod and c.key_down(0x54)):
            self.cycle_theme()
        if self._edge("up", mod and c.key_down(0x26)):
            self.change_mass(0.1)
        if self._edge("down", mod and c.key_down(0x28)):
            self.change_mass(-0.1)

    def spawn(self, kind, fling=None):
        if len(self.bodies) >= MAX_BODIES:
            self.bodies.pop(0)
        self.bodies.append(Body(kind, fling))

    def handle_clicks(self, now):
        for kind, pressed in self.clicks.poll():
            if self.mode != 0:
                self._hold.clear()
                self._flick.clear()
                continue
            if pressed:
                self._hold[kind] = now
                if self.clicks.key_down(0x12):               # Alt held: wait for the release and flick
                    p = QCursor.pos()
                    self._flick[kind] = (p.x(), p.y())
                else:
                    self.spawn(kind)
            else:
                self._hold.pop(kind, None)
                start = self._flick.pop(kind, None)
                if start is not None:
                    p = QCursor.pos()
                    dx, dy = p.x() - start[0], p.y() - start[1]
                    dist = math.hypot(dx, dy)
                    if dist < 15:
                        self.spawn(kind)
                    else:
                        self.spawn(kind, fling=(math.atan2(-dy, dx), min(dist / 250.0, 1.0)))

    # ---------------- fullscreen pause ----------------
    def _check_fullscreen(self):
        try:
            fs = foreground_is_fullscreen()
        except Exception:
            fs = False
        if fs and not self.paused:
            self.paused = True
            self._cursor_was_hidden = _cursor_hidden
            self.hide()
            if self._cursor_was_hidden:
                restore_system_cursor()          # give the game its normal cursor back
        elif not fs and self.paused:
            self.paused = False
            self.show()
            self.exclude_from_capture()
            if self._cursor_was_hidden:
                hide_system_cursor()

    # ---------------- main loop ----------------
    def tick(self):
        now = time.perf_counter()
        dt = min(now - self._last, 0.05)
        self._last = now

        self.poll_hotkeys()
        if self.cfg["pause_on_fullscreen"] and now - self._fs_t > 0.5:
            self._fs_t = now
            self._check_fullscreen()
        if self.paused:
            self.clicks.poll()
            return

        self.handle_clicks(now)

        # gravity well: holding a button pulls everything in faster
        held = [now - t for t in self._hold.values()]
        target_pull = 1.0 if (held and max(held) > 0.35) else 0.0
        self.pull += (target_pull - self.pull) * min(1.0, dt * 8.0)
        pull_factor = 1.0 + 2.0 * self.pull

        alive = []
        for b in self.bodies:
            if b.step(dt, pull_factor):
                alive.append(b)
            else:
                # swallowed: the hole flares, rings like a bell and (for stars) flashes spikes
                self.flare = min(1.5, self.flare + BODY_FLARE[b.kind])
                self.spike = min(1.5, self.spike + BODY_SPIKE[b.kind])
                self.wave = 0.0
        self.bodies = alive

        # mouse motion, idle detection
        pos = QCursor.pos()
        cur = (pos.x(), pos.y())
        if self._last_cursor is not None and dt > 0:
            vx, vy = (cur[0] - self._last_cursor[0]) / dt, (cur[1] - self._last_cursor[1]) / dt
            a = min(1.0, dt * 15.0)
            self._vel = (self._vel[0] * (1 - a) + vx * a, self._vel[1] * (1 - a) + vy * a)
        if cur != self._last_cursor:
            self._last_move = now
        self._last_cursor = cur
        speed = math.hypot(*self._vel)
        if speed > 20.0:
            self._mdir = (self._vel[0] / speed, -self._vel[1] / speed)
        self._mamt += (min(speed / 3000.0, 1.0) - self._mamt) * min(1.0, dt * 10.0)

        idle = now - self._last_move
        k = _clamp((idle - self.cfg["idle_seconds"]) / 3.0, 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)

        # decays
        self.flare = max(self.flare * math.exp(-dt * 1.2), 0.0)
        flare_floor = max(0.35 * k, 0.4 * self.pull)
        if self.flare < flare_floor:
            self.flare = flare_floor
        self.spike *= math.exp(-dt * 2.5)
        self.wave = min(self.wave + dt, 99.0)

        base = self.cfg["mass"] * (0.8 if self.mode == 2 else 1.0)
        target = _clamp(base * (1.0 + 0.12 * k), 0.6, 1.2)
        self.hole_scale += (target - self.hole_scale) * min(1.0, dt * 6.0)

        # keep the overlay centred on the real cursor (the hole centre is the click point)
        gx, gy = pos.x() - self.SIZE // 2, pos.y() - self.SIZE // 2
        if (gx, gy) != self._last_pos:
            self.move(gx, gy)
            self._last_pos = (gx, gy)

        self._frame_n += 1
        if self.lens_ok and (not self.cfg["low_power"] or self._frame_n % 2 == 0):
            try:
                frame = self.grab_backdrop(gx, gy)
            except Exception:
                frame = None
            if frame is not None:
                self._frame = frame
        self.update()


# ----------------------------------------------------------------------------------------------
# tray icon
# ----------------------------------------------------------------------------------------------
def build_tray(app, overlay):
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None

    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 140, 30))
    p.drawEllipse(4, 4, 56, 56)
    p.setBrush(QColor(255, 220, 150))
    p.drawEllipse(10, 10, 44, 44)
    p.setBrush(QColor(0, 0, 0))
    p.drawEllipse(18, 18, 28, 28)
    p.end()

    tray = QSystemTrayIcon(QIcon(pm), app)
    tray.setToolTip("Black hole cursor")
    menu = QMenu()

    act_cursor = QAction("Show real cursor", menu)
    act_cursor.setCheckable(True)
    act_cursor.triggered.connect(lambda _=False: overlay.toggle_cursor())
    menu.addAction(act_cursor)

    act_low = QAction("Low power mode", menu)
    act_low.setCheckable(True)
    act_low.triggered.connect(lambda _=False: overlay.toggle_low_power())
    menu.addAction(act_low)

    mode_menu = menu.addMenu("Mode")
    mode_group = QActionGroup(mode_menu)
    mode_actions = []
    for i, name in enumerate(MODE_NAMES):
        a = QAction(name, mode_menu)
        a.setCheckable(True)
        a.triggered.connect(lambda _=False, i=i: overlay.set_mode(i))
        mode_group.addAction(a)
        mode_menu.addAction(a)
        mode_actions.append(a)

    theme_menu = menu.addMenu("Theme")
    theme_group = QActionGroup(theme_menu)
    theme_actions = {}
    for name in THEMES:
        a = QAction(name.capitalize(), theme_menu)
        a.setCheckable(True)
        a.triggered.connect(lambda _=False, n=name: overlay.set_theme(n))
        theme_group.addAction(a)
        theme_menu.addAction(a)
        theme_actions[name] = a

    menu.addSeparator()
    act_quit = QAction("Quit", menu)
    act_quit.triggered.connect(lambda _=False: QApplication.quit())
    menu.addAction(act_quit)

    def refresh():
        act_cursor.setChecked(not _cursor_hidden)
        act_low.setChecked(bool(overlay.cfg["low_power"]))
        for i, a in enumerate(mode_actions):
            a.setChecked(i == overlay.mode)
        for n, a in theme_actions.items():
            a.setChecked(n == overlay.theme)

    tray.setContextMenu(menu)
    tray._menu = menu           # keep references alive
    tray._refresh = refresh
    overlay.on_change = refresh
    refresh()
    tray.show()
    return tray


if __name__ == "__main__":
    cfg = load_config()

    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)      # hiding the overlay (fullscreen pause) must not quit
    # Ctrl+C must go through a normal quit so the real cursor gets restored
    signal.signal(signal.SIGINT, lambda *a: app.quit())
    app.aboutToQuit.connect(restore_system_cursor)

    overlay = BlackHoleCursor(cfg)
    app.aboutToQuit.connect(lambda: save_config(overlay.cfg))
    overlay.show()
    overlay.exclude_from_capture()
    if overlay.lens_ok and cfg["hide_cursor"]:
        hide_system_cursor()
    tray = build_tray(app, overlay)
    if tray is not None:
        tray._refresh()
    sys.exit(app.exec_())