/*
 * Minimal INT8 TensorFlow Lite interpreter for openmv/expression.tflite.
 *
 * Supports exactly the operators that model uses: CONV_2D, MAX_POOL_2D,
 * RESHAPE, FULLY_CONNECTED and SOFTMAX, with per-channel weight scales.
 * It runs in the browser (and in Node for testing) with no dependencies,
 * so the presentation page never needs a TFLite runtime or a network fetch.
 *
 * Usage:
 *   const model = EmpathModel.load(arrayBuffer);
 *   const scores = model.predictGray(gray48);       // Uint8Array(48*48), 0..255
 *   const map = model.occlusion(gray48, 6);         // { base, top, drops[36] }
 */
(function (root) {
  'use strict';

  const OP = { ADD: 0, AVERAGE_POOL_2D: 1, CONV_2D: 3, DEPTHWISE_CONV_2D: 4, FULLY_CONNECTED: 9, MAX_POOL_2D: 17, RESHAPE: 22, SOFTMAX: 25, PAD: 34, MEAN: 40 };
  const TYPE = { FLOAT32: 0, INT32: 2, INT8: 9 };

  // ---- FlatBuffer reader (only what the TFLite schema needs) ----
  function reader(buf) {
    const dv = new DataView(buf);
    const u16 = o => dv.getUint16(o, true), u32 = o => dv.getUint32(o, true), i32 = o => dv.getInt32(o, true);
    const field = (t, i) => {
      const vt = t - i32(t), size = u16(vt), o = 4 + 2 * i;
      if (o >= size) return 0;
      const off = u16(vt + o);
      return off ? t + off : 0;
    };
    const vec = (t, i) => { const p = field(t, i); if (!p) return null; const v = p + u32(p); return { len: u32(v), at: v + 4 }; };
    return {
      root: u32(0),
      table(t, i) { const p = field(t, i); return p ? p + u32(p) : 0; },
      tables(t, i) { const v = vec(t, i); return v ? Array.from({ length: v.len }, (_, k) => { const e = v.at + 4 * k; return e + u32(e); }) : []; },
      ints(t, i) { const v = vec(t, i); return v ? Array.from({ length: v.len }, (_, k) => i32(v.at + 4 * k)) : []; },
      floats(t, i) { const v = vec(t, i); return v ? Array.from({ length: v.len }, (_, k) => dv.getFloat32(v.at + 4 * k, true)) : []; },
      longs(t, i) { const v = vec(t, i); return v ? Array.from({ length: v.len }, (_, k) => Number(dv.getBigInt64(v.at + 8 * k, true))) : []; },
      bytes(t, i) { const v = vec(t, i); return v ? { at: v.at, len: v.len } : null; },
      i8(t, i, def) { const p = field(t, i); return p ? dv.getInt8(p) : def; },
      i32(t, i, def) { const p = field(t, i); return p ? i32(p) : def; },
      u32(t, i, def) { const p = field(t, i); return p ? u32(p) : def; },
      f32(t, i, def) { const p = field(t, i); return p ? dv.getFloat32(p, true) : def; },
      dv
    };
  }

  function parse(buf) {
    const R = reader(buf), m = R.root;
    const codes = R.tables(m, 1).map(c => Math.max(R.i8(c, 0, 0), R.i32(c, 3, 0)));
    const buffers = R.tables(m, 4).map(b => R.bytes(b, 0));
    const sg = R.tables(m, 2)[0];
    const tensors = R.tables(sg, 0).map(t => {
      const q = R.table(t, 4);
      const info = {
        shape: R.ints(t, 0), type: R.i8(t, 1, 0), buffer: R.u32(t, 2, 0),
        scale: q ? R.floats(q, 2) : [], zp: q ? R.longs(q, 3) : [], data: null
      };
      const b = buffers[info.buffer];
      if (b && b.len) {
        if (info.type === TYPE.INT8) info.data = new Int8Array(buf.slice(b.at, b.at + b.len));
        else if (info.type === TYPE.INT32) { const n = b.len / 4, a = new Int32Array(n); for (let k = 0; k < n; k++) a[k] = R.dv.getInt32(b.at + 4 * k, true); info.data = a; }
        else if (info.type === TYPE.FLOAT32) { const n = b.len / 4, a = new Float32Array(n); for (let k = 0; k < n; k++) a[k] = R.dv.getFloat32(b.at + 4 * k, true); info.data = a; }
      }
      return info;
    });
    const ops = R.tables(sg, 3).map(o => {
      const opt = R.table(o, 4), code = codes[R.u32(o, 0, 0)];
      const op = { code, ins: R.ints(o, 1), outs: R.ints(o, 2) };
      if (code === OP.CONV_2D) Object.assign(op, { pad: R.i8(opt, 0, 0), sw: R.i32(opt, 1, 1), sh: R.i32(opt, 2, 1), act: R.i8(opt, 3, 0) });
      else if (code === OP.DEPTHWISE_CONV_2D) Object.assign(op, { pad: R.i8(opt, 0, 0), sw: R.i32(opt, 1, 1), sh: R.i32(opt, 2, 1), mult: R.i32(opt, 3, 1), act: R.i8(opt, 4, 0) });
      else if (code === OP.MAX_POOL_2D || code === OP.AVERAGE_POOL_2D) Object.assign(op, { pad: R.i8(opt, 0, 0), sw: R.i32(opt, 1, 1), sh: R.i32(opt, 2, 1), fw: R.i32(opt, 3, 1), fh: R.i32(opt, 4, 1), act: R.i8(opt, 5, 0) });
      else if (code === OP.FULLY_CONNECTED) Object.assign(op, { act: opt ? R.i8(opt, 0, 0) : 0 });
      else if (code === OP.SOFTMAX) Object.assign(op, { beta: opt ? R.f32(opt, 0, 1) : 1 });
      else if (code === OP.ADD) Object.assign(op, { act: opt ? R.i8(opt, 0, 0) : 0 });
      else if (code !== OP.RESHAPE && code !== OP.PAD && code !== OP.MEAN) throw new Error('Unsupported operator code ' + code);
      return op;
    });
    return { tensors, ops, input: R.ints(sg, 1)[0], output: R.ints(sg, 2)[0] };
  }

  // ---- quantization helpers ----
  const clamp8 = v => v < -128 ? -128 : v > 127 ? 127 : v;
  const roundHalfAway = v => (v < 0 ? -Math.round(-v) : Math.round(v));
  const actMin = (act, s, z) => act === 1 || act === 3 ? z : -128;          // RELU / RELU6 floor at zero
  const actMax = (act, s, z) => act === 3 ? Math.min(127, z + Math.round(6 / s)) : 127;
  const pads = (n, k, stride, same) => {
    if (!same) return { out: Math.floor((n - k) / stride) + 1, before: 0 };
    const out = Math.ceil(n / stride), total = Math.max((out - 1) * stride + k - n, 0);
    return { out, before: Math.floor(total / 2) };
  };

  function conv(op, T, G) {
    const xt = G.tensors[op.ins[0]], wt = G.tensors[op.ins[1]], bt = op.ins[2] >= 0 ? G.tensors[op.ins[2]] : null, yt = G.tensors[op.outs[0]];
    const x = T[op.ins[0]], w = wt.data, bias = bt ? bt.data : null;
    const [, H, W, C] = xt.shape, [O, KH, KW] = wt.shape;
    const py = pads(H, KH, op.sh, op.pad === 0), px = pads(W, KW, op.sw, op.pad === 0), OH = py.out, OW = px.out;
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0;
    const lo = actMin(op.act, sy, zy), hi = actMax(op.act, sy, zy);
    const mult = new Float64Array(O); for (let o = 0; o < O; o++) mult[o] = sx * (wt.scale.length > 1 ? wt.scale[o] : wt.scale[0]) / sy;
    const y = new Int8Array(OH * OW * O);
    for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
      const ob = (oy * OW + ox) * O;
      for (let o = 0; o < O; o++) {
        let acc = bias ? bias[o] : 0;
        for (let ky = 0; ky < KH; ky++) {
          const iy = oy * op.sh - py.before + ky; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < KW; kx++) {
            const ix = ox * op.sw - px.before + kx; if (ix < 0 || ix >= W) continue;
            const xb = (iy * W + ix) * C, wb = ((o * KH + ky) * KW + kx) * C;
            for (let c = 0; c < C; c++) acc += (x[xb + c] - zx) * w[wb + c];
          }
        }
        const q = roundHalfAway(acc * mult[o]) + zy;
        y[ob + o] = q < lo ? lo : q > hi ? hi : q;
      }
    }
    T[op.outs[0]] = y;
  }

  function maxpool(op, T, G) {
    const xt = G.tensors[op.ins[0]], x = T[op.ins[0]], [, H, W, C] = xt.shape;
    const py = pads(H, op.fh, op.sh, op.pad === 0), px = pads(W, op.fw, op.sw, op.pad === 0);
    const y = new Int8Array(py.out * px.out * C);
    for (let oy = 0; oy < py.out; oy++) for (let ox = 0; ox < px.out; ox++) for (let c = 0; c < C; c++) {
      let m = -128;
      for (let ky = 0; ky < op.fh; ky++) {
        const iy = oy * op.sh - py.before + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < op.fw; kx++) {
          const ix = ox * op.sw - px.before + kx; if (ix < 0 || ix >= W) continue;
          const v = x[(iy * W + ix) * C + c]; if (v > m) m = v;
        }
      }
      y[(oy * px.out + ox) * C + c] = m;
    }
    T[op.outs[0]] = y;
  }

  function depthwise(op, T, G) {
    const xt = G.tensors[op.ins[0]], wt = G.tensors[op.ins[1]], bt = op.ins[2] >= 0 ? G.tensors[op.ins[2]] : null, yt = G.tensors[op.outs[0]];
    const x = T[op.ins[0]], w = wt.data, bias = bt ? bt.data : null;
    const [, H, W, C] = xt.shape, [, KH, KW, OC] = wt.shape, mult = op.mult || OC / C;
    const py = pads(H, KH, op.sh, op.pad === 0), px = pads(W, KW, op.sw, op.pad === 0), OH = py.out, OW = px.out;
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0;
    const lo = actMin(op.act, sy, zy), hi = actMax(op.act, sy, zy);
    const m = new Float64Array(OC); for (let o = 0; o < OC; o++) m[o] = sx * (wt.scale.length > 1 ? wt.scale[o] : wt.scale[0]) / sy;
    const y = new Int8Array(OH * OW * OC);
    for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) for (let ic = 0; ic < C; ic++) for (let k = 0; k < mult; k++) {
      const o = ic * mult + k; let acc = bias ? bias[o] : 0;
      for (let ky = 0; ky < KH; ky++) {
        const iy = oy * op.sh - py.before + ky; if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < KW; kx++) {
          const ix = ox * op.sw - px.before + kx; if (ix < 0 || ix >= W) continue;
          acc += (x[(iy * W + ix) * C + ic] - zx) * w[(ky * KW + kx) * OC + o];
        }
      }
      const q = roundHalfAway(acc * m[o]) + zy;
      y[(oy * OW + ox) * OC + o] = q < lo ? lo : q > hi ? hi : q;
    }
    T[op.outs[0]] = y;
  }

  function add(op, T, G) {
    const at = G.tensors[op.ins[0]], bt = G.tensors[op.ins[1]], yt = G.tensors[op.outs[0]];
    const a = T[op.ins[0]] || at.data, b = T[op.ins[1]] || bt.data;
    const za = at.zp[0] || 0, sa = at.scale[0], zb = bt.zp[0] || 0, sb = bt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0;
    const lo = actMin(op.act, sy, zy), hi = actMax(op.act, sy, zy), n = Math.max(a.length, b.length), y = new Int8Array(n);
    for (let k = 0; k < n; k++) {
      const q = roundHalfAway(((a[k % a.length] - za) * sa + (b[k % b.length] - zb) * sb) / sy) + zy;
      y[k] = q < lo ? lo : q > hi ? hi : q;
    }
    T[op.outs[0]] = y;
  }

  function pad(op, T, G) {
    const xt = G.tensors[op.ins[0]], x = T[op.ins[0]], p = G.tensors[op.ins[1]].data, [, H, W, C] = xt.shape;
    const top = p[2], bottom = p[3], left = p[4], right = p[5], OH = H + top + bottom, OW = W + left + right, z = xt.zp[0] || 0;
    const y = new Int8Array(OH * OW * C).fill(z);
    for (let iy = 0; iy < H; iy++) y.set(x.subarray(iy * W * C, (iy + 1) * W * C), ((iy + top) * OW + left) * C);
    T[op.outs[0]] = y;
  }

  function mean(op, T, G) {   // global average over height and width (axes 1, 2), as produced by GlobalAveragePooling2D
    const xt = G.tensors[op.ins[0]], yt = G.tensors[op.outs[0]], x = T[op.ins[0]], [, H, W, C] = xt.shape;
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0, y = new Int8Array(C);
    for (let c = 0; c < C; c++) { let s = 0; for (let k = 0; k < H * W; k++) s += x[k * C + c] - zx; y[c] = clamp8(roundHalfAway(s / (H * W) * sx / sy) + zy); }
    T[op.outs[0]] = y;
  }

  function avgpool(op, T, G) {
    const xt = G.tensors[op.ins[0]], yt = G.tensors[op.outs[0]], x = T[op.ins[0]], [, H, W, C] = xt.shape;
    const py = pads(H, op.fh, op.sh, op.pad === 0), px = pads(W, op.fw, op.sw, op.pad === 0);
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0, y = new Int8Array(py.out * px.out * C);
    for (let oy = 0; oy < py.out; oy++) for (let ox = 0; ox < px.out; ox++) for (let c = 0; c < C; c++) {
      let s = 0, n = 0;
      for (let ky = 0; ky < op.fh; ky++) { const iy = oy * op.sh - py.before + ky; if (iy < 0 || iy >= H) continue; for (let kx = 0; kx < op.fw; kx++) { const ix = ox * op.sw - px.before + kx; if (ix < 0 || ix >= W) continue; s += x[(iy * W + ix) * C + c] - zx; n++; } }
      y[(oy * px.out + ox) * C + c] = clamp8(roundHalfAway(s / n * sx / sy) + zy);
    }
    T[op.outs[0]] = y;
  }

  function fully(op, T, G) {
    const xt = G.tensors[op.ins[0]], wt = G.tensors[op.ins[1]], bt = op.ins[2] >= 0 ? G.tensors[op.ins[2]] : null, yt = G.tensors[op.outs[0]];
    const x = T[op.ins[0]], w = wt.data, bias = bt ? bt.data : null, [O, N] = wt.shape;
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0;
    const lo = actMin(op.act, sy, zy), hi = actMax(op.act, sy, zy), y = new Int8Array(O);
    for (let o = 0; o < O; o++) {
      let acc = bias ? bias[o] : 0; const wb = o * N;
      for (let n = 0; n < N; n++) acc += (x[n] - zx) * w[wb + n];
      const q = roundHalfAway(acc * sx * (wt.scale.length > 1 ? wt.scale[o] : wt.scale[0]) / sy) + zy;
      y[o] = q < lo ? lo : q > hi ? hi : q;
    }
    T[op.outs[0]] = y;
  }

  function softmax(op, T, G) {
    const xt = G.tensors[op.ins[0]], yt = G.tensors[op.outs[0]], x = T[op.ins[0]];
    const zx = xt.zp[0] || 0, sx = xt.scale[0], sy = yt.scale[0], zy = yt.zp[0] || 0;
    const r = Array.from(x, v => (v - zx) * sx * op.beta), mx = Math.max(...r), e = r.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
    T[op.outs[0]] = Int8Array.from(e, v => clamp8(roundHalfAway(v / s / sy) + zy));
  }

  function load(buf) {
    const G = parse(buf), it = G.tensors[G.input], ot = G.tensors[G.output];
    const [, H, W, C] = it.shape, side = H;
    if (H !== W || C !== 1 || it.type !== TYPE.INT8) throw new Error('Expected a square single-channel INT8 input');
    const fcOp = G.ops.find(op => op.code === OP.FULLY_CONNECTED), featIndex = fcOp ? fcOp.ins[0] : -1;
    let lastT = null;
    const run = q => {
      const T = new Array(G.tensors.length); T[G.input] = q; lastT = T;
      for (const op of G.ops) {
        if (op.code === OP.CONV_2D) conv(op, T, G);
        else if (op.code === OP.DEPTHWISE_CONV_2D) depthwise(op, T, G);
        else if (op.code === OP.MAX_POOL_2D) maxpool(op, T, G);
        else if (op.code === OP.AVERAGE_POOL_2D) avgpool(op, T, G);
        else if (op.code === OP.RESHAPE) T[op.outs[0]] = T[op.ins[0]];
        else if (op.code === OP.FULLY_CONNECTED) fully(op, T, G);
        else if (op.code === OP.SOFTMAX) softmax(op, T, G);
        else if (op.code === OP.ADD) add(op, T, G);
        else if (op.code === OP.PAD) pad(op, T, G);
        else if (op.code === OP.MEAN) mean(op, T, G);
      }
      return T[G.output];
    };
    // 48×48 grey -> side×side grey with TensorFlow's bilinear resize (half-pixel centres), matching training's upsample().
    const fit = gray => {
      const n = Math.round(Math.sqrt(gray.length)); if (n === side) return gray;
      const out = new Uint8Array(side * side), s = n / side;
      for (let y = 0; y < side; y++) {
        const fy = Math.min(Math.max((y + .5) * s - .5, 0), n - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, n - 1), wy = fy - y0;
        for (let x = 0; x < side; x++) {
          const fx = Math.min(Math.max((x + .5) * s - .5, 0), n - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, n - 1), wx = fx - x0;
          const v = (gray[y0 * n + x0] * (1 - wx) + gray[y0 * n + x1] * wx) * (1 - wy) + (gray[y1 * n + x0] * (1 - wx) + gray[y1 * n + x1] * wx) * wy;
          out[y * side + x] = Math.min(255, Math.max(0, Math.round(v)));
        }
      }
      return out;
    };
    const sIn = it.scale[0], zIn = it.zp[0] || 0, sOut = ot.scale[0], zOut = ot.zp[0] || 0;
    const quantize = gray => Int8Array.from(gray, g => clamp8(roundHalfAway(g / 255 / sIn) + zIn));
    const scores = q => Array.from(run(q), v => (v - zOut) * sOut);
    return {
      side,
      runInt8: run,
      // gray: 48*48 values 0..255 (row-major), same as the training images and OpenMV's grayscale ROI.
      // Models with a larger input (MobileNetV2 96×96) get it upsampled exactly as in training.
      predictGray(gray) { return scores(quantize(fit(gray))); },
      // Same as predictGray, plus the dequantized feature vector feeding the final layer (for per-person prototypes).
      predictWithFeatures(gray) {
        const s = scores(quantize(fit(gray))), ft = G.tensors[featIndex], raw = lastT[featIndex];
        const f = new Float32Array(raw.length), z = ft.zp[0] || 0, sc = ft.scale[0];
        for (let k = 0; k < raw.length; k++) f[k] = (raw[k] - z) * sc;
        return { scores: s, features: f };
      },
      // Occlusion map: cover each cell of a grid×grid layout with the image's mean grey,
      // rerun the model, and record how much the top class's score drops.
      occlusion(gray48, grid = 6) {
        const base = scores(quantize(fit(gray48))), top = base.indexOf(Math.max(...base));
        const n = Math.round(Math.sqrt(gray48.length)), mean = gray48.reduce((a, b) => a + b, 0) / gray48.length, cell = n / grid, drops = new Float32Array(grid * grid);
        const work = Uint8Array.from(gray48);
        for (let gy = 0; gy < grid; gy++) for (let gx = 0; gx < grid; gx++) {
          work.set(gray48);
          for (let y = Math.round(gy * cell); y < Math.round((gy + 1) * cell); y++) for (let x = Math.round(gx * cell); x < Math.round((gx + 1) * cell); x++) work[y * n + x] = mean;
          drops[gy * grid + gx] = Math.max(0, base[top] - scores(quantize(fit(work)))[top]);
        }
        return { base, top, drops, grid };
      }
    };
  }

  root.EmpathModel = { load };
})(typeof window !== 'undefined' ? window : globalThis);
