/*
 * The Inside Out "brows" logo, which wears the expression being read.
 * Shapes are in a 100x100 box: two brows (quadratic curves) and two lens-shaped eyes.
 * openmv/make_brows_icons.py draws the OLED icons from the same numbers; keep the two in sync.
 *
 *   const s = Brows.shape('angry');   // { l, r, el, er } SVG path strings
 */
(function (root) {
  // Eye: a lens around (cx, 58 + dy), half-width hw; top/bottom are the control-point offsets of its two curves.
  const eye = (cx, hw, top, bottom, dy) => {
    const y = 58 + dy;
    return `M${cx - hw} ${y} Q${cx} ${y + top} ${cx + hw} ${y} Q${cx} ${y + bottom} ${cx - hw} ${y} Z`;
  };
  const SHAPES = {
    happy:    { l: 'M18 40 Q31 28 44 36', r: 'M56 36 Q69 28 82 40', el: [7, -13, -5, 2], er: [7, -13, -5, 2] },
    angry:    { l: 'M16 32 Q30 37 44 44', r: 'M56 44 Q70 37 84 32', el: [6, -5, 11, 1], er: [6, -5, 11, 1] },
    surprise: { l: 'M18 30 Q31 13 44 27', r: 'M56 27 Q69 13 82 30', el: [7, -16, 16, 0], er: [7, -16, 16, 0] },
    neutral:  { l: 'M18 38 Q31 38 44 38', r: 'M56 38 Q69 38 82 38', el: [6, -12, 12, 0], er: [6, -12, 12, 0] },
    sad:      { l: 'M18 40 Q31 38 44 31', r: 'M56 31 Q69 38 82 40', el: [6, -7, 10, 2], er: [6, -7, 10, 2] },
    unsure:   { l: 'M18 39 Q31 37 44 39', r: 'M56 31 Q69 19 82 29', el: [6, -10, 10, 1], er: [6, -13, 13, 0] }
  };
  const ALIAS = { fear: 'surprise', disgust: 'angry' };
  const shape = mood => {
    const s = SHAPES[mood] || SHAPES[ALIAS[mood]] || SHAPES.neutral;
    return { l: s.l, r: s.r, el: eye(31, ...s.el), er: eye(69, ...s.er) };
  };
  root.Brows = { shape, SHAPES, ALIAS };
})(typeof self !== 'undefined' ? self : this);
