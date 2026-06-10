/** cross_opt_v1 + score — funziona solo con ultima cinquina (client-side) */

function comp(n) {
  if (n === 45 || n === 90) return null;
  const c = 90 - n;
  return c >= 1 && c <= 90 ? c : null;
}

function vert(n) {
  if (n < 10 || n > 90) return null;
  const s = String(n);
  if (s.length !== 2) return null;
  const rev = parseInt(s.split("").reverse().join(""), 10);
  if (rev === n || rev < 1 || rev > 90) return null;
  return rev;
}

function isTwin(n) {
  return n >= 10 && n <= 88 && n % 11 === 0;
}

function neighbors(n, twin11) {
  const step = twin11 && isTwin(n) ? 11 : 1;
  const out = [];
  for (const d of [-step, step]) {
    const x = n + d;
    if (x >= 1 && x <= 90) out.push(x);
  }
  return out;
}

function adjDiffs(nums) {
  const d = [];
  for (let i = 0; i < nums.length - 1; i++) {
    d.push(Math.max(nums[i], nums[i + 1]) - Math.min(nums[i], nums[i + 1]));
  }
  return d;
}

function add(pool, n) {
  if (n != null && n >= 1 && n <= 90) pool.add(n);
}

function poolCrossOptV1(nums) {
  const pool = new Set();
  const diffs = adjDiffs(nums);
  for (const n of nums) {
    const cn = comp(n);
    for (const d of diffs) {
      add(pool, n + d);
      add(pool, Math.abs(n - d));
      if (cn != null) {
        add(pool, cn + d);
        add(pool, Math.abs(cn - d));
      }
      for (const base of [n, cn, vert(n)].filter(Boolean)) {
        for (const x of neighbors(base, true)) add(pool, x);
      }
    }
    add(pool, comp(n));
    add(pool, vert(n));
  }
  for (const d of diffs) {
    const vd = vert(d);
    if (vd) for (const x of neighbors(vd, true)) add(pool, x);
  }
  return pool;
}

function scoreCross(nums) {
  const scores = {};
  const bump = (n, w) => {
    if (n != null && n >= 1 && n <= 90) scores[n] = (scores[n] || 0) + w;
  };
  const diffs = adjDiffs(nums);
  for (const n of nums) {
    const cn = comp(n);
    for (const d of diffs) {
      bump(n + d, 2);
      bump(Math.abs(n - d), 1);
      if (cn != null) {
        bump(cn + d, 4);
        bump(Math.abs(cn - d), 2);
      }
      for (const base of [n, cn, vert(n)].filter(Boolean)) {
        for (const x of neighbors(base, true)) bump(x, isTwin(base) ? 2 : 1);
      }
    }
  }
  if (nums.includes(45)) bump(45, 3);
  if (nums.includes(90)) bump(9, 2);
  return scores;
}

function topN(pool, scores, n) {
  return [...pool]
    .map((num) => [num, scores[num] || 0])
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, n)
    .map((x) => x[0]);
}

function bestPair(scores, cands) {
  let best = [cands[0], cands[1]];
  let bestS = -1;
  for (let i = 0; i < cands.length; i++) {
    for (let j = i + 1; j < cands.length; j++) {
      const s = (scores[cands[i]] || 0) + (scores[cands[j]] || 0);
      if (s > bestS) {
        bestS = s;
        best = [cands[i], cands[j]];
      }
    }
  }
  return best;
}

function comboBest(scores, cands, k) {
  if (cands.length < k) return cands.slice();
  let best = cands.slice(0, k);
  let bestS = -1;
  function rec(start, chosen) {
    if (chosen.length === k) {
      const s = chosen.reduce((a, n) => a + (scores[n] || 0), 0);
      if (s > bestS) {
        bestS = s;
        best = [...chosen];
      }
      return;
    }
    for (let i = start; i < cands.length; i++) {
      chosen.push(cands[i]);
      rec(i + 1, chosen);
      chosen.pop();
    }
  }
  rec(0, []);
  return best;
}

export function predictCrossOnly(quintina) {
  const pool = poolCrossOptV1(quintina);
  const scores = scoreCross(quintina);
  const top8 = topN(pool, scores, 8);
  const top12 = topN(pool, scores, 12);
  const m = Math.min(16, top12.length);
  const cands = topN(pool, scores, m);
  return {
    pool_size: pool.size,
    pool_top12: top12,
    ambo: bestPair(scores, top8.length >= 2 ? top8 : top12),
    terno: comboBest(scores, cands, 3),
    quaterna: comboBest(scores, cands, 4),
    cinquina: comboBest(scores, cands, 5),
    mode: "cross",
  };
}
