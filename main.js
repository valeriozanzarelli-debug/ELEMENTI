/* main.js (server) - porting of Wix homeVelocode.js
   Requirements:
   - Custom Elements emit CustomEvent("ic-message", { detail: {...} })
   - We send to CE by setting attributes: ic_in (json) and ic_in_id (change triggers)
   - Tracking: window.dataLayer (GTM) optional
   - Portfolio API: POST /api/portfolio  { mode, selection, limit, gallery } -> { items: [] }
*/

"use strict";

/* ===========================
   1) CONFIG / SELECTOR
=========================== */
const CE_HERO_ID = "#hero";
const CE_PREVENTIVO_ID = "#preventivo";
const CE_GALLERIA_ID = "#galleria";
const CE_PRENOTAZIONI_ID = "#prenotazioni";
const CE_METODO_ID = "#metodo";
const CE_COSTA_CALMA_ID = "#costacalma";
const CE_CONTATTI_ID = "#contatti";

// Anchors (IDs in DOM)
const ANCHOR_STIMA_ID = "#AnchorStima";
const ANCHOR_METODO_ID = "#AnchorMetodo";
const ANCHOR_PRENOTAZIONE_ID = "#AnchorPrenotazione";

/* ===========================
   2) BUSINESS
=========================== */
const WHATSAPP_OWNER_E164_DIGITS = "34659316793";
const PRIVACY_URL = "https://www.iubenda.com/privacy-policy/54166793";
const ALLOWED_DOW = [1, 3, 5];
const GTM_ADS_CONVERSION_EVENT = "lead_submit";

/* ===========================
   3) SECTION TIMING MAP
=========================== */
let sectionTimers = {};
const SECTIONS_MAP = {
  [CE_HERO_ID]: "hero",
  [CE_PREVENTIVO_ID]: "preventivo",
  [CE_GALLERIA_ID]: "galleria",
  [CE_METODO_ID]: "metodo",
  [CE_PRENOTAZIONI_ID]: "prenotazioni",
  [CE_COSTA_CALMA_ID]: "costa_calma",
  [CE_CONTATTI_ID]: "contatti"
};

/* ===========================
   4) RUNTIME STATE
=========================== */
let currentSelection = null;
let currentFavorites = [];
let leadSubmitFired = false;
let sessionNonce = "";

let __lastPortfolioCountByGallery = { done: 0, available: 0 };

let __portfolioPrefetchStarted = false;
let __galleriaReady = false;
let __portfolioPrefetch = {
  done: { items: null, limit: 8, ts: 0 },
  available: { items: null, limit: 8, ts: 0 }
};

let __portfolioSignalLastTsByKey = {};
let _scrollLastY = null;
let _scrollDir = "down";
let _scrollTicking = false;
let _lastSection = "";

let _selLastSig = "";
let _selLastTs = 0;

let __sessionEndFired = false;

let __prevAreasSet = new Set();
let __areasClicksTotal = 0;

let sessionState = {
  entry: null,
  sectionsViewed: {},
  sectionTime: {},
  sectionActions: {},
  totalTimeSeconds: 0,
  highIntentActionsCount: 0,
  eventsCount: 0,
  lastActiveTs: Date.now(),
  _didSelectionChange: false,
  _didPrenotaClick: false,
  _didWhatsappClick: false,
  lead_saved_ok: false
};

/* ===========================
   5) UTILS
=========================== */
function $(sel) {
  return document.querySelector(sel);
}

function safeStr(v, max = 120) {
  if (v === null || v === undefined) return "";
  const s = String(v).replace(/[\u0000-\u001F\u007F]/g, "").trim();
  return s.length > max ? s.slice(0, max) : s;
}
function safeNumOrNull(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function normalizeActionName(name) {
  let s = safeStr(name, 80).toLowerCase();
  if (!s) return "";
  s = s.replace(/[^a-z0-9_]/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  if (s.length > 40) s = s.slice(0, 40);
  return s;
}
function extractAreaValue(item) {
  try {
    if (item === null || item === undefined) return "";
    if (typeof item === "string" || typeof item === "number" || typeof item === "boolean") {
      return safeStr(item, 60);
    }
    if (typeof item === "object") {
      const keys = ["id", "name", "value", "slug", "label", "key", "title"];
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        if (item[k] !== undefined && item[k] !== null) {
          const v = safeStr(item[k], 60);
          if (v) return v;
        }
      }
      try { return safeStr(JSON.stringify(item), 60); } catch (_) { return ""; }
    }
    return "";
  } catch (_) {
    return "";
  }
}
function normalizeAreaKey(v) {
  let s = safeStr(v, 60).toLowerCase();
  if (!s) return "";
  s = s.replace(/[^a-z0-9_]/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  if (s.length > 40) s = s.slice(0, 40);
  return s;
}
function normalizeSelectionForPortfolio(sel) {
  const out = { areas: [], price: { min: null, max: null, mode: "range" } };
  if (!sel || typeof sel !== "object") return out;

  if (sel.areas && Array.isArray(sel.areas)) {
    const a = [];
    for (let i = 0; i < sel.areas.length; i++) {
      const raw = extractAreaValue(sel.areas[i]);
      const x = safeStr(raw, 60);
      if (x) a.push(x);
      if (a.length >= 10) break;
    }
    out.areas = a;
  }

  if (sel.price && typeof sel.price === "object") {
    let min = safeNumOrNull(sel.price.min);
    let max = safeNumOrNull(sel.price.max);
    if (min !== null && max !== null && min > max) [min, max] = [max, min];
    out.price = { mode: safeStr(sel.price.mode, 20) || "range", min, max };
  }
  return out;
}
function compactSelectionParams(sel) {
  const n = normalizeSelectionForPortfolio(sel || {});
  const areas = Array.isArray(n.areas) ? n.areas.slice(0, 6) : [];
  return {
    areas: areas.join(","),
    areas_count: areas.length,
    price_mode: safeStr(n.price?.mode || "range", 20),
    price_min: n.price?.min != null ? Number(n.price.min) : null,
    price_max: n.price?.max != null ? Number(n.price.max) : null
  };
}
function emptySelection() {
  return { price: { mode: "empty" }, areas: [], rawAreas: [] };
}
function normalizeSectionForState(section) {
  let s = safeStr(section, 40).toLowerCase();
  if (!s || s === "home") return "";
  s = s.replace(/[^a-z0-9_]/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  if (s.length > 40) s = s.slice(0, 40);
  return s;
}
function getDeviceHint() {
  // simple heuristic
  const w = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
  if (w <= 768) return "mobile";
  if (w <= 1024) return "tablet";
  return "desktop";
}
const SUPPORTED_LANGS = ["it", "en", "de", "es"];
function getLangPage() {
  try {
    const host = String(window.location.hostname || "").toLowerCase();
    const sub = host.split(".")[0];
    return SUPPORTED_LANGS.includes(sub) ? sub : "en";
  } catch (_) {
    return "en";
  }
}
function getQueryParams() {
  const out = {};
  try {
    const sp = new URLSearchParams(window.location.search || "");
    sp.forEach((v, k) => { out[String(k)] = v; });
  } catch (_) {}
  return out;
}
function buildEntryUrlSafe() {
  const pathOnly = safeStr(window.location.pathname || "/", 200) || "/";
  const qp = getQueryParams();
  const keys = ["utm_source","utm_medium","utm_campaign","utm_content","utm_term","gclid"];
  const kept = [];
  keys.forEach((k) => {
    const v = qp[k];
    if (v !== undefined && v !== null && String(v).trim() !== "") {
      kept.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(v)));
    }
  });
  const urlOut = pathOnly + (kept.length ? "?" + kept.join("&") : "");
  return safeStr(urlOut, 500);
}
function makeNonce() {
  try {
    return `${Date.now()}_${Math.random().toString(16).slice(2)}_${Math.random().toString(16).slice(2)}`;
  } catch (_) {
    return String(Date.now());
  }
}
function makeEventId() {
  try { return `${Date.now()}_${Math.random().toString(16).slice(2)}`; }
  catch (_) { return String(Date.now()); }
}
function initEntryContext() {
  if (sessionState.entry) return sessionState.entry;
  const qp = getQueryParams();
  sessionState.entry = {
    utm_source: safeStr(qp.utm_source, 100),
    utm_medium: safeStr(qp.utm_medium, 100),
    utm_campaign: safeStr(qp.utm_campaign, 120),
    utm_content: safeStr(qp.utm_content, 120),
    utm_term: safeStr(qp.utm_term, 120),
    gclid: safeStr(qp.gclid, 140),
    entry_url: buildEntryUrlSafe(),
    entry_page: safeStr(window.location.pathname || "/", 120)
  };
  return sessionState.entry;
}

/* ===========================
   5B) WhatsApp helpers
=========================== */
function buildWhatsAppLeadId() {
  return safeStr(sessionNonce, 160);
}
function buildWhatsAppPrefillText(sectionTag) {
  const sec = safeStr(sectionTag || "home", 40) || "home";
  const lang = safeStr(getLangPage(), 10) || "en";
  return `lead: ${buildWhatsAppLeadId()}\nsection: ${sec}\nlang: ${lang}`;
}
function urlHasParam(url, param) {
  try { return String(url || "").includes(param + "="); } catch (_) { return false; }
}
function appendTextParam(url, text) {
  try {
    const u = safeStr(url, 900);
    if (!u) return u;
    if (urlHasParam(u, "text")) return u;
    const enc = encodeURIComponent(String(text || ""));
    if (!enc) return u;
    return u.includes("?") ? (u + "&text=" + enc) : (u + "?text=" + enc);
  } catch (_) {
    return safeStr(url, 900);
  }
}
function normalizeWhatsAppBaseUrl(rawUrl) {
  const u = safeStr(rawUrl, 900);
  if (!u) return "https://wa.me/" + WHATSAPP_OWNER_E164_DIGITS;
  return u;
}

/* ===========================
   6) TRACKING -> dataLayer
=========================== */
const EVENT_PARAM_ALLOWLIST = {
  ic_home_ready: ["section"],
  ic_section_ready: ["section", "component"],
  ic_section_nav: ["from", "to", "dir"],
  ic_section_exit: ["section", "seconds", "dir", "section_actions"],
  ic_session_end: [
    "exit_reason","last_section","total_time_seconds","section_time_map",
    "events_count","high_intent_actions_count","areas_count"
  ],
  ic_click_scroll: ["from", "to", "trigger"],
  ic_open_privacy: ["from", "url", "trigger"],
  ic_open_url: ["from", "url"],
  ic_whatsapp_click: ["section", "url"],
  ic_prenota_click: ["section", "source"],
  ic_selection_changed: ["from", "areas", "areas_count", "price_mode", "price_min", "price_max"],
  ic_portfolio_request: ["from","mode","limit","gallery","areas","areas_count","price_mode","price_min","price_max"],
  ic_portfolio_results: ["from", "count", "mode", "gallery"],
  ic_portfolio_error: ["from", "error"],
  ic_likes_changed: ["count"],
  ic_lead_submit: ["section", "source"]
};

function flatParams(params) {
  const p = params && typeof params === "object" ? params : {};
  const out = {};
  Object.keys(p).forEach((k) => {
    const key = safeStr(k, 60);
    if (!key) return;
    const v = p[k];
    if (v && typeof v === "object") {
      try { out[key] = JSON.stringify(v).slice(0, 220); }
      catch (_) { out[key] = safeStr(v, 120); }
    } else {
      out[key] = v;
    }
  });
  return out;
}
function filterParamsForAction(action, params) {
  const allow = EVENT_PARAM_ALLOWLIST[action];
  if (!allow || !Array.isArray(allow)) {
    const minimal = {};
    ["section","from","to","source","ok","count","seconds","dir","url"].forEach((k) => {
      if (params && params[k] !== undefined) minimal[k] = params[k];
    });
    return minimal;
  }
  const out = {};
  allow.forEach((k) => {
    if (!params || params[k] === undefined) return;
    out[k] = params[k];
  });
  return out;
}
function isHighIntentAction(action) {
  const a = String(action || "");
  if (!a) return false;
  if (a.startsWith("ic_whatsapp_click")) return true;
  if (a.startsWith("ic_prenota_click")) return true;
  if (a.startsWith("ic_lead_submit")) return true;
  return false;
}
function enrichPayload(p, actionName) {
  if (!p || typeof p !== "object") return p;
  const entry = initEntryContext();

  if (p.session_id === undefined) p.session_id = safeStr(sessionNonce, 120);
  if (p.event_id === undefined) p.event_id = safeStr(makeEventId(), 80);
  if (p.event_ts === undefined) p.event_ts = Date.now();

  if (p.lang_page === undefined) p.lang_page = safeStr(getLangPage(), 10);
  if (p.device_hint === undefined) p.device_hint = safeStr(getDeviceHint(), 10);
  if (p.page_type === undefined) p.page_type = "home";

  if (p.utm_source === undefined) p.utm_source = entry.utm_source;
  if (p.utm_medium === undefined) p.utm_medium = entry.utm_medium;
  if (p.utm_campaign === undefined) p.utm_campaign = entry.utm_campaign;
  if (p.utm_content === undefined) p.utm_content = entry.utm_content;
  if (p.utm_term === undefined) p.utm_term = entry.utm_term;
  if (p.gclid === undefined) p.gclid = entry.gclid;
  if (p.entry_url === undefined) p.entry_url = entry.entry_url;
  if (p.entry_page === undefined) p.entry_page = entry.entry_page;

  sessionState.eventsCount = (sessionState.eventsCount || 0) + 1;
  sessionState.lastActiveTs = Date.now();

  const sectionMaybe = p.section || p.from || p.component || "";
  const s = normalizeSectionForState(sectionMaybe);
  if (s) {
    sessionState.sectionsViewed[s] = true;
    sessionState.sectionActions[s] = (Number(sessionState.sectionActions[s]) || 0) + 1;
  }
  if (isHighIntentAction(actionName)) {
    sessionState.highIntentActionsCount = (sessionState.highIntentActionsCount || 0) + 1;
  }

  Object.keys(p).forEach((k) => {
    if (typeof p[k] === "string") p[k] = safeStr(p[k], 500);
  });

  return p;
}
function dlPush(actionName, params) {
  const action = normalizeActionName(actionName);
  if (!action) return;

  const raw = flatParams(params);
  const p = filterParamsForAction(action, raw);

  const pagePath = safeStr(window.location.pathname || "/", 120) || "/";
  if (p.page === undefined) p.page = pagePath;

  if (p.section === undefined || p.section === null || String(p.section).trim() === "") {
    if (p.from) p.section = p.from;
    else if (p.component) p.section = p.component;
    else p.section = "home";
  }

  p.action = action;
  p.event = "inkconscious_event";

  enrichPayload(p, action);

  // GTM
  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push({ event: p.event, detail: p });
}
function dlPushConversion(eventName, params) {
  const evName = safeStr(eventName, 60);
  if (!evName) return;

  const p = flatParams(params);
  const pagePath = safeStr(window.location.pathname || "/", 120) || "/";
  if (p.page === undefined) p.page = pagePath;

  if (p.section === undefined || p.section === null || String(p.section).trim() === "") {
    if (p.from) p.section = p.from;
    else p.section = "home";
  }

  p.event = evName;
  enrichPayload(p, evName);

  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push(p); // conversion event as root object (common GTM pattern)
}

/* ===========================
   7) SECTION TIMING + SCROLL
=========================== */
function addSectionTime(section, seconds) {
  const s = normalizeSectionForState(section);
  if (!s) return;
  const sec = Number(seconds) || 0;
  if (sec <= 0) return;
  sessionState.sectionTime[s] = (Number(sessionState.sectionTime[s]) || 0) + sec;
}
function bindScrollDirection() {
  window.addEventListener("scroll", () => {
    if (_scrollTicking) return;
    _scrollTicking = true;
    setTimeout(() => {
      _scrollTicking = false;
      try {
        const y = window.scrollY || 0;
        if (_scrollLastY === null || _scrollLastY === undefined) { _scrollLastY = y; return; }
        if (y > _scrollLastY + 6) _scrollDir = "down";
        else if (y < _scrollLastY - 6) _scrollDir = "up";
        _scrollLastY = y;
      } catch (_) {}
    }, 120);
  }, { passive: true });
}
function onSectionEnter(sectionName) {
  const s = normalizeSectionForState(sectionName);
  if (!s) return;
  if (_lastSection && _lastSection !== s) {
    dlPush("ic_section_nav", { from: _lastSection, to: s, dir: _scrollDir });
  }
  _lastSection = s;
}
function onSectionExit(sectionName, durationSec) {
  const s = normalizeSectionForState(sectionName);
  if (!s) return;
  const sec = Number(durationSec) || 0;
  if (sec <= 0) return;

  sessionState.totalTimeSeconds = (sessionState.totalTimeSeconds || 0) + sec;
  addSectionTime(s, sec);

  const sectionActions = Number(sessionState.sectionActions[s] || 0);
  dlPush("ic_section_exit", { section: s, seconds: sec, dir: _scrollDir, section_actions: sectionActions });
}
function sectionTimeMapCompact() {
  try {
    const st = sessionState.sectionTime || {};
    const keys = Object.keys(st).sort();
    const parts = [];
    for (let i = 0; i < keys.length; i++) {
      const k = normalizeSectionForState(keys[i]);
      if (!k) continue;
      const v = Math.max(0, Math.round(Number(st[keys[i]]) || 0));
      if (!v) continue;
      parts.push(k + ":" + String(v));
    }
    return parts.join("|");
  } catch (_) { return ""; }
}
function finalizeSessionOnce(reason) {
  try {
    if (__sessionEndFired) return;
    __sessionEndFired = true;
    dlPush("ic_session_end", {
      exit_reason: safeStr(reason || "unknown", 40),
      last_section: safeStr(_lastSection || "home", 40),
      total_time_seconds: Math.max(0, Math.round(Number(sessionState.totalTimeSeconds || 0))),
      section_time_map: sectionTimeMapCompact(),
      events_count: Number(sessionState.eventsCount || 0),
      high_intent_actions_count: Number(sessionState.highIntentActionsCount || 0),
      areas_count: Math.max(0, Math.round(Number(__areasClicksTotal || 0)))
    });
  } catch (_) {}
}
function bindExitHooks() {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") finalizeSessionOnce("visibility_hidden");
  });
  window.addEventListener("pagehide", () => finalizeSessionOnce("pagehide"), { capture: true });
}

// IntersectionObserver replacement for Wix viewport enter/leave
function initTimingTrackers() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const el = entry.target;
      const id = "#" + el.id;
      const section = SECTIONS_MAP[id];
      if (!section) return;

      if (entry.isIntersecting) {
        sectionTimers[id] = Date.now();
        onSectionEnter(section);
      } else {
        const start = sectionTimers[id];
        if (!start) return;
        const duration = Math.round((Date.now() - start) / 1000);
        if (duration >= 3) onSectionExit(section, duration);
        sectionTimers[id] = null;
      }
    });
  }, { threshold: 0.25 });

  Object.keys(SECTIONS_MAP).forEach((idSel) => {
    const el = $(idSel);
    if (!el) return;
    obs.observe(el);
  });
}

/* ===========================
   8) CUSTOM ELEMENTS - Send/Receive
=========================== */
function withNonce(payload) {
  if (!payload || typeof payload !== "object") return payload;
  return { ...payload, _nonce: sessionNonce };
}
function ceSend(selector, payload) {
  try {
    const el = $(selector);
    if (!el || typeof el.setAttribute !== "function") return;
    const msg = withNonce(payload || {});
    el.setAttribute("ic_in", JSON.stringify(msg));
    el.setAttribute("ic_in_id", makeEventId());
  } catch (_) {}
}
function getBaseConfigForCE(sectionTag) {
  const tag = safeStr(sectionTag || "home", 40) || "home";
  return {
    type: "config",
    nonce: sessionNonce,
    lang: safeStr(getLangPage(), 10),
    whatsappOwnerDigits: WHATSAPP_OWNER_E164_DIGITS,
    privacyUrl: PRIVACY_URL,
    allowedDow: ALLOWED_DOW,
    whatsappLeadId: buildWhatsAppLeadId(),
    whatsappPrefillText: buildWhatsAppPrefillText(tag)
  };
}
function applyResize(selector, height, minH, maxH) {
  try {
    const h = Number(height);
    if (!Number.isFinite(h) || h <= 0) return;
    const clamped = Math.max(minH, Math.min(maxH, Math.round(h)));
    const el = $(selector);
    if (el) el.style.height = clamped + "px";
  } catch (_) {}
}
function wireCE(selector, handler, tag) {
  const el = $(selector);
  if (!el) return;

  el.addEventListener("ic-message", (event) => {
    try {
      const msg = event && event.detail ? event.detail : null;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "ready") {
        ceSend(selector, getBaseConfigForCE(tag || selector));
        dlPush("ic_section_ready", { section: tag || selector, component: tag || selector });
      }

      if (msg.type === "resize" && msg.height) {
        if (selector === CE_HERO_ID) applyResize(CE_HERO_ID, msg.height, 420, 1600);
        if (selector === CE_METODO_ID) applyResize(CE_METODO_ID, msg.height, 320, 2600);
        if (selector === CE_COSTA_CALMA_ID) applyResize(CE_COSTA_CALMA_ID, msg.height, 320, 1800);
        if (selector === CE_PREVENTIVO_ID) applyResize(CE_PREVENTIVO_ID, msg.height, 360, 2400);
        if (selector === CE_GALLERIA_ID) applyResize(CE_GALLERIA_ID, msg.height, 360, 3200);
        if (selector === CE_PRENOTAZIONI_ID) applyResize(CE_PRENOTAZIONI_ID, msg.height, 420, 2600);
        if (selector === CE_CONTATTI_ID) applyResize(CE_CONTATTI_ID, msg.height, 360, 2000);
      }

      handler(msg, selector, tag);
    } catch (_) {}
  });
}

/* ===========================
   9) UI helpers
=========================== */
function scrollToElement(selector) {
  const el = $(selector);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}
function scrollToAnchorId(anchorName) {
  let a = safeStr(anchorName, 80);
  if (!a) return;
  if (!a.startsWith("#")) a = "#" + a;
  scrollToElement(a);
}

/* ===========================
   10) NAV comuni
=========================== */
function handleCommonNav(msg, fromTag) {
  const from = fromTag || "home";
  const trigger = safeStr(msg?.trigger || "", 60) || undefined;

  if (msg.type === "scrollToStima") {
    dlPush("ic_click_scroll", { from, to: "stima", trigger });
    scrollToElement(ANCHOR_STIMA_ID);
    return true;
  }
  if (msg.type === "scrollToMetodo") {
    dlPush("ic_click_scroll", { from, to: "metodo", trigger });
    scrollToElement(ANCHOR_METODO_ID);
    return true;
  }
  if (msg.type === "scrollToPrenotazione" || msg.type === "scrollToPrenotazioni") {
    dlPush("ic_click_scroll", { from, to: "prenotazione", trigger });
    scrollToElement(ANCHOR_PRENOTAZIONE_ID);
    return true;
  }
  if (msg.type === "scrollToGalleria" || msg.type === "scrollToGallery") {
    dlPush("ic_click_scroll", { from, to: "galleria" });
    scrollToElement(CE_GALLERIA_ID);
    return true;
  }
  if (msg.type === "scrollToAnchor" && msg.anchor) {
    dlPush("ic_click_scroll", { from, to: safeStr(msg.anchor, 80), trigger });
    scrollToAnchorId(msg.anchor);
    return true;
  }

  if (msg.type === "openWhatsApp") {
    sessionState._didWhatsappClick = true;
    const sectionTag = safeStr(from || "home", 40) || "home";
    const base = normalizeWhatsAppBaseUrl(msg.url || msg.whatsappUrl || "") || ("https://wa.me/" + WHATSAPP_OWNER_E164_DIGITS);
    const u = appendTextParam(base, buildWhatsAppPrefillText(sectionTag));
    dlPush("ic_whatsapp_click", { section: sectionTag, url: u });
    // apertura la fa il CE, qui solo tracking
    return true;
  }

  if (msg.type === "openPrivacy") {
    dlPush("ic_open_privacy", { from, url: PRIVACY_URL, trigger: trigger || "" });
    return true;
  }

  return false;
}

/* ===========================
   11) HANDLERS
=========================== */
function onHeroMessage(msg) { if (handleCommonNav(msg, "hero")) return; }
function onMetodoMessage(msg) { if (handleCommonNav(msg, "metodo")) return; }
function onCostaCalmaMessage(msg) { if (handleCommonNav(msg, "costa_calma")) return; }
function onContattiMessage(msg) {
  if (handleCommonNav(msg, "contatti")) return;
  if (msg.type === "openUrl" && msg.url) {
    dlPush("ic_open_url", { from: "contatti", url: safeStr(msg.url, 900) });
  }
}

/* ===========================
   12) PORTFOLIO (server)
=========================== */
function normalizeGalleryKey(raw) {
  const g = safeStr(raw, 20).toLowerCase();
  return g === "available" ? "available" : "done";
}
function shouldDebouncePortfolioSignal(type, gallery) {
  try {
    const t = safeStr(type, 40);
    const g = normalizeGalleryKey(gallery);
    const key = t + "|" + g;
    const now = Date.now();
    const last = Number(__portfolioSignalLastTsByKey[key] || 0);
    if (last && now - last < 250) return true;
    __portfolioSignalLastTsByKey[key] = now;
    return false;
  } catch (_) {
    return false;
  }
}
function getPrefetchCacheIfUsable(gallery, mode, limit, selectionOrNull) {
  try {
    if (selectionOrNull) return null;
    if (String(mode || "").toLowerCase() !== "default") return null;

    const g = normalizeGalleryKey(gallery);
    const lim = Number(limit) || 8;
    const c = __portfolioPrefetch[g];
    if (!c) return null;
    if (Number(c.limit || 0) !== lim) return null;
    if (!Array.isArray(c.items)) return null;
    return c.items;
  } catch (_) {
    return null;
  }
}

async function apiGetPortfolio(payload) {
  // Default: POST /api/portfolio
  const res = await fetch("/api/portfolio", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload || {})
  });
  if (!res.ok) throw new Error("portfolio_http_" + res.status);
  return await res.json();
}

function startPortfolioPrefetchOnce() {
  if (__portfolioPrefetchStarted) return;
  __portfolioPrefetchStarted = true;

  const limit = Number(__portfolioPrefetch.done.limit) || 8;

  ["done", "available"].forEach((g) => {
    apiGetPortfolio({ mode: "default", selection: null, limit, gallery: g })
      .then((out) => {
        const items = out && Array.isArray(out.items) ? out.items : [];
        __portfolioPrefetch[g] = { items, limit, ts: Date.now() };

        if (__galleriaReady && g === "done") {
          __lastPortfolioCountByGallery.done = items.length;
          ceSend(CE_GALLERIA_ID, { type: "portfolioResults", items, source: "home", from: "prefetch" });
        }
      })
      .catch(() => {});
  });
}

function trackPortfolioRequestOnce(fromTag, gallery, mode, limit, selectionOrNull) {
  const compact = selectionOrNull
    ? compactSelectionParams(selectionOrNull)
    : compactSelectionParams(currentSelection || {});
  dlPush("ic_portfolio_request", {
    from: fromTag || "",
    mode,
    limit,
    gallery,
    areas: compact.areas,
    areas_count: compact.areas_count,
    price_mode: compact.price_mode,
    price_min: compact.price_min,
    price_max: compact.price_max
  });
}
function trackPortfolioResultsOnTabClick(fromTag, gallery, mode, explicitCountOrNull) {
  const count =
    explicitCountOrNull != null && Number.isFinite(Number(explicitCountOrNull))
      ? Number(explicitCountOrNull)
      : Number(__lastPortfolioCountByGallery[gallery] || 0);

  dlPush("ic_portfolio_results", { from: fromTag || "", count, mode: mode || "default", gallery });
}

function handleRequestPortfolio(msg, fromTag) {
  const mode =
    String(msg.mode || "").toLowerCase() === "default"
      ? "default"
      : msg.selection ? "filtered" : "default";

  const limit = Number(msg.limit) || 8;
  const selection = msg.selection ? normalizeSelectionForPortfolio(msg.selection) : null;
  const gallery = normalizeGalleryKey(msg.gallery);

  const cached = getPrefetchCacheIfUsable(gallery, mode, limit, selection);
  if (cached) {
    __lastPortfolioCountByGallery[gallery] = cached.length;
    ceSend(CE_GALLERIA_ID, { type: "portfolioResults", items: cached, source: "home", from: fromTag || "" });
    return Promise.resolve({ ok: true, cached: true });
  }

  return apiGetPortfolio({ mode, selection, limit, gallery })
    .then((out) => {
      const items = out && Array.isArray(out.items) ? out.items : [];
      __lastPortfolioCountByGallery[gallery] = items.length;
      ceSend(CE_GALLERIA_ID, { type: "portfolioResults", items, source: "home", from: fromTag || "" });
    })
    .catch((e) => {
      dlPush("ic_portfolio_error", { from: fromTag || "", error: safeStr(String(e?.message || e), 180) });
      ceSend(CE_GALLERIA_ID, { type: "portfolioResults", items: [], error: String(e?.message || e), source: "home" });
    });
}

/* set diff helper */
function symmetricDiffSets(aSet, bSet) {
  const out = [];
  try {
    aSet.forEach((v) => { if (!bSet.has(v)) out.push(v); });
    bSet.forEach((v) => { if (!aSet.has(v)) out.push(v); });
  } catch (_) {}
  return out;
}
function buildAreasKeySetFromSelection(selObj) {
  const n = normalizeSelectionForPortfolio(selObj || {});
  const areas = Array.isArray(n.areas) ? n.areas : [];
  const s = new Set();
  for (let i = 0; i < areas.length; i++) {
    const k = normalizeAreaKey(areas[i]);
    if (k) s.add(k);
  }
  return { set: s, normalized: n };
}

function onPreventivoMessage(msg) {
  if (handleCommonNav(msg, "preventivo")) return;

  if (msg.type === "selectionChanged" && msg.selection) {
    sessionState._didSelectionChange = true;
    currentSelection = msg.selection;

    const built = buildAreasKeySetFromSelection(msg.selection);
    const newSet = built.set;
    const nsel = built.normalized;

    const sorted = Array.from(newSet).sort();
    const sig =
      sorted.join("|") +
      "|" + safeStr(nsel.price?.mode || "range", 20) +
      "|" + String(nsel.price?.min != null ? nsel.price.min : "") +
      "|" + String(nsel.price?.max != null ? nsel.price.max : "");

    const now = Date.now();
    if (sig === _selLastSig && now - _selLastTs < 250) {
      ceSend(CE_PRENOTAZIONI_ID, { type: "syncSelection", selection: msg.selection });
      return;
    }
    _selLastSig = sig;
    _selLastTs = now;

    const toggled = symmetricDiffSets(__prevAreasSet, newSet);
    if (toggled && toggled.length) {
      const price_mode = safeStr(nsel.price?.mode || "range", 20);
      const price_min = nsel.price?.min != null ? Number(nsel.price.min) : null;
      const price_max = nsel.price?.max != null ? Number(nsel.price.max) : null;

      for (let i = 0; i < toggled.length; i++) {
        const part = normalizeAreaKey(toggled[i]);
        if (!part) continue;
        __areasClicksTotal = (Number(__areasClicksTotal) || 0) + 1;
        dlPush("ic_selection_changed", {
          from: "preventivo",
          areas: part,
          areas_count: __areasClicksTotal,
          price_mode,
          price_min,
          price_max
        });
      }
    }

    __prevAreasSet = newSet;
    ceSend(CE_PRENOTAZIONI_ID, { type: "syncSelection", selection: msg.selection });
    return;
  }

  if (msg.type === "requestPortfolio") {
    handleRequestPortfolio(msg, "preventivo");
    return;
  }

  if (msg.type === "scrollToGallery") {
    dlPush("ic_click_scroll", { from: "preventivo", to: "galleria" });
    scrollToElement(CE_GALLERIA_ID);
  }
}

function onGalleriaMessage(msg) {
  if (handleCommonNav(msg, "galleria")) return;

  if (msg.type === "ready") {
    __galleriaReady = true;
    try {
      const c = __portfolioPrefetch.done;
      if (c && Array.isArray(c.items)) {
        __lastPortfolioCountByGallery.done = c.items.length;
        ceSend(CE_GALLERIA_ID, { type: "portfolioResults", items: c.items, source: "home", from: "prefetch" });
      }
    } catch (_) {}
    return;
  }

  if (msg.type === "likesChanged" && Array.isArray(msg.likes)) {
    currentFavorites = msg.likes;
    dlPush("ic_likes_changed", { count: msg.likes.length });
    return;
  }

  if (msg.type === "requestSyncSelection") {
    const sel = currentSelection || emptySelection();
    ceSend(CE_GALLERIA_ID, { type: "syncSelection", selection: sel });
    ceSend(CE_PRENOTAZIONI_ID, { type: "syncSelection", selection: sel });
    ceSend(CE_GALLERIA_ID, { type: "syncLikes", likes: Array.isArray(currentFavorites) ? currentFavorites : [] });
    return;
  }

  if (msg.type === "requestPortfolio") {
    handleRequestPortfolio(msg, "galleria");
    return;
  }

  if (msg.type === "portfolioPopupOpen") {
    const galleryPop = normalizeGalleryKey(msg.gallery);
    if (shouldDebouncePortfolioSignal("portfolioPopupOpen", galleryPop)) return;

    const modePop = safeStr(msg.mode || "default", 20).toLowerCase() === "filtered" ? "filtered" : "default";
    const limitPop = Number(msg.limit) || 8;
    const selPop = currentSelection ? normalizeSelectionForPortfolio(currentSelection) : null;
    trackPortfolioRequestOnce("galleria", galleryPop, modePop, limitPop, selPop);
    return;
  }

  if (msg.type === "portfolioTabClick") {
    const galleryTab = normalizeGalleryKey(msg.gallery);
    if (shouldDebouncePortfolioSignal("portfolioTabClick", galleryTab)) return;

    const modeTab = safeStr(msg.mode || "default", 20).toLowerCase() === "filtered" ? "filtered" : "default";
    const countTab = msg.count != null ? Number(msg.count) : null;
    trackPortfolioResultsOnTabClick("galleria", galleryTab, modeTab, countTab);
    return;
  }

  if (msg.type === "resetSelection") {
    currentSelection = null;
    __prevAreasSet = new Set();
    __areasClicksTotal = 0;
    _selLastSig = "";
    _selLastTs = 0;

    ceSend(CE_PREVENTIVO_ID, { type: "syncSelection", selection: emptySelection() });
    ceSend(CE_PRENOTAZIONI_ID, { type: "syncSelection", selection: emptySelection() });
    return;
  }

  if (msg.type === "resetPreventivo") {
    ceSend(CE_PREVENTIVO_ID, { type: "reset" });
  }
}

/* ===========================
   13) PRENOTAZIONI
=========================== */
function fireAdsLeadOnce(source, section, extraParams) {
  if (leadSubmitFired) return;
  leadSubmitFired = true;

  const p = { ...(extraParams && typeof extraParams === "object" ? extraParams : {}) };
  p.source = safeStr(source, 40) || "unknown";
  p.section = safeStr(section, 40) || "home";

  dlPushConversion(GTM_ADS_CONVERSION_EVENT, p);
}

function onPrenotazioniMessage(msg, selector, tag) {
  if (handleCommonNav(msg, tag)) return;

  if (msg.type === "ready") {
    ceSend(selector, {
      type: "config",
      nonce: sessionNonce,
      lang: safeStr(getLangPage(), 10),
      whatsappOwnerDigits: WHATSAPP_OWNER_E164_DIGITS,
      privacyUrl: PRIVACY_URL,
      allowedDow: ALLOWED_DOW,
      whatsappLeadId: buildWhatsAppLeadId(),
      whatsappPrefillText: buildWhatsAppPrefillText(tag)
    });
    return;
  }

  // calendly disabled
  if (String(msg.type || "").startsWith("calendly")) {
    if (msg.type === "calendlyCheckDay") {
      ceSend(selector, { type: "calendlyAvailability", dateISO: msg.dateISO, available: false, error: "disabled", times: [], reqId: msg.reqId || "" });
    }
    if (msg.type === "calendlyCreateLink") {
      ceSend(selector, { type: "calendlyLink", bookingUrl: "", error: "disabled" });
    }
    if (msg.type === "calendlyBookInvitee") {
      ceSend(selector, { type: "calendlyBooked", eventUri: "", error: "disabled" });
    }
    return;
  }

  if (msg.type === "saveLead") {
    dlPush("ic_lead_submit", { section: tag, source: safeStr(msg.source || tag, 30) });
    fireAdsLeadOnce("lead_form", tag, { ok: true, source: safeStr(msg.source || tag, 30) });
    ceSend(selector, { type: "leadSaved", ok: true, id: "", error: "" });
    return;
  }

  if (msg.type === "requestSyncSelection") {
    ceSend(selector, { type: "syncSelection", selection: currentSelection || emptySelection() });
  }
}

/* ===========================
   15) INIT
=========================== */
function init() {
  sessionNonce = makeNonce();

  startPortfolioPrefetchOnce();

  bindScrollDirection();
  initTimingTrackers();
  bindExitHooks();

  wireCE(CE_HERO_ID, onHeroMessage, "hero");
  wireCE(CE_METODO_ID, onMetodoMessage, "metodo");
  wireCE(CE_COSTA_CALMA_ID, onCostaCalmaMessage, "costa_calma");
  wireCE(CE_PREVENTIVO_ID, onPreventivoMessage, "preventivo");
  wireCE(CE_GALLERIA_ID, onGalleriaMessage, "galleria");

  setTimeout(() => { ceSend(CE_GALLERIA_ID, getBaseConfigForCE("galleria")); }, 250);

  wireCE(CE_PRENOTAZIONI_ID, (msg) => onPrenotazioniMessage(msg, CE_PRENOTAZIONI_ID, "prenotazioni"), "prenotazioni");
  wireCE(CE_CONTATTI_ID, onContattiMessage, "contatti");

  dlPush("ic_home_ready", { section: "home" });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}