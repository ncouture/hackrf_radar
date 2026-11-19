(function () {
  'use strict';
  var KEY = 'hackrf-radar-color-scheme';
  var stored = null;
  try {
    stored = window.localStorage.getItem(KEY);
  } catch (_) {
    stored = null;
  }
  var meta = document.querySelector('meta[name="color-scheme"]');
  if (stored === 'light' || stored === 'dark') {
    if (meta) meta.setAttribute('content', stored);
    document.documentElement.setAttribute('data-theme', stored);
  } else {
    if (meta) meta.setAttribute('content', 'light dark');
    document.documentElement.removeAttribute('data-theme');
  }
})();
