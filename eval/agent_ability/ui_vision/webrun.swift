// Render and drive one offline task page that follows the benchmark contract (window.benchmark.reset(mode) and
// getState(), data-item-index on click targets, data-field-label on fill controls), for real.py.
//
//   swift webrun.swift <job.json>
//
// job.json: {"page": "<page.html>", "width": W, "height": H,
//            "renders": [{"mode": "click|fill|read", "out": "<out-prefix>"}],
//            "replays": [{"key": "...", "mode": "...", "actions": [{"kind": "click|type|select", "x": px, "y": px,
//                         "text": "..."}]}],
//            "replay_out": "<file.jsonl>"}
// A render resets the page to its mode and writes <out>.png (W x H, scale 1) and <out>.elements.json (the "ax" list
// of render.swift plus the boxes of data-item-index and data-field-label elements). A replay resets the page to its
// mode, reads getState() as "before", performs the actions at their points (a click clicks the element there; a
// type or select sets the native control there, or the control of the label there, and presses anything else),
// and writes "before", "after"
// and a per-action log as one JSON line. Navigation away from the page is blocked and the page scrolls back to the
// top before every step, so each item starts from the same view. Off screen; nothing appears on the display.

import AppKit
import WebKit

let args = CommandLine.arguments
guard args.count == 2, let jobData = FileManager.default.contents(atPath: args[1]),
      let job = try? JSONSerialization.jsonObject(with: jobData) as? [String: Any],
      let pagePath = job["page"] as? String, let width = job["width"] as? Double, let height = job["height"] as? Double else {
    FileHandle.standardError.write("usage: swift webrun.swift <job.json>\n".data(using: .utf8)!)
    exit(2)
}
let page = URL(fileURLWithPath: pagePath).standardizedFileURL
let viewport = NSSize(width: width, height: height)
let renders = job["renders"] as? [[String: Any]] ?? []
let replays = job["replays"] as? [[String: Any]] ?? []
let replayOut = job["replay_out"] as? String

func js(_ value: Any) -> String {
    // A JSON value as a JavaScript expression, passed through base64 so no quoting can break the script.
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed])
    return "JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('\(data.base64EncodedString())'), c => c.charCodeAt(0))))"
}

let reset = "(m => { scrollTo(0, 0); if (window.benchmark) window.benchmark.reset(m); scrollTo(0, 0); return 1; })"

let collect = """
(() => {
  const box = el => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
  const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
      && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; };
  const text = el => (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
  const name = el => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.getAttribute('aria-labelledby')) return el.getAttribute('aria-labelledby').split(' ')
      .map(id => document.getElementById(id)).filter(Boolean).map(text).join(' ');
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) return text(l); }
    const wrap = el.closest('label'); if (wrap && wrap !== el) return text(wrap);
    if (el.placeholder) return el.placeholder;
    if (el.title) return el.title;
    if (el.alt) return el.alt;
    return text(el).slice(0, 120);
  };
  const role = el => {
    const r = el.getAttribute('role'); if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'input') { const ty = (el.type || 'text').toLowerCase();
      return {checkbox: 'checkbox', radio: 'radio', submit: 'button', button: 'button', file: 'button',
              range: 'slider'}[ty] || 'textbox'; }
    return {a: 'link', button: 'button', select: 'combobox', textarea: 'textbox', label: 'label',
            h1: 'heading', h2: 'heading', h3: 'heading', h4: 'heading', th: 'columnheader', td: 'cell',
            img: 'image'}[t] || t;
  };
  const sel = 'input, select, textarea, button, a, [role], label, h1, h2, h3, h4, th, td, img[alt]';
  const ax = [];
  document.querySelectorAll('body *').forEach(el => {
    if (!visible(el) || (el.type || '').toLowerCase() === 'hidden') return;
    if (!el.matches(sel)) {
      const own = Array.from(el.childNodes).filter(n => n.nodeType === Node.TEXT_NODE)
        .map(n => n.textContent).join(' ').replace(/\\s+/g, ' ').trim();
      if (own && !el.closest(sel)) ax.push({role: 'text', name: own.slice(0, 200), box: box(el)});
      return;
    }
    const item = {role: role(el), name: name(el), box: box(el)};
    if (el.placeholder && el.placeholder !== item.name) item.placeholder = el.placeholder;
    if ('value' in el && el.tagName !== 'BUTTON' && el.value && !['checkbox', 'radio'].includes(el.type)) item.value = el.value;
    if (el.tagName === 'SELECT') item.value = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : '';
    if (['checkbox', 'radio'].includes(el.type)) item.checked = el.checked;
    if (el.getAttribute('aria-checked')) item.checked = el.getAttribute('aria-checked') === 'true';
    ax.push(item);
  });
  const items = {}, fields = {};
  document.querySelectorAll('[data-item-index]').forEach(el => {
    (items[el.dataset.itemIndex] = items[el.dataset.itemIndex] || []).push({box: box(el), visible: visible(el)}); });
  document.querySelectorAll('[data-field-label]').forEach(el => {
    (fields[el.dataset.fieldLabel] = fields[el.dataset.fieldLabel] || []).push(
      {box: box(el), visible: visible(el), tag: el.tagName.toLowerCase(), type: el.type || ''}); });
  const groups = {};
  document.querySelectorAll('[data-field-group]').forEach(el => {
    (groups[el.dataset.fieldGroup] = groups[el.dataset.fieldGroup] || []).push({box: box(el), visible: visible(el),
      value: el.value && el.value !== 'on' ? el.value : el.getAttribute('aria-label') || ''}); });
  return JSON.stringify({ax, items, fields, groups, scrollHeight: document.documentElement.scrollHeight,
                         state: window.benchmark ? window.benchmark.getState() : null});
})()
"""

let replayJS = """
(job => {
  scrollTo(0, 0);
  const B = window.benchmark;
  if (B) B.reset(job.mode);
  scrollTo(0, 0);
  const before = B ? B.getState() : null;
  const norm = s => String(s == null ? '' : s).replace(/\\s+/g, ' ').trim().toLowerCase();
  const control = el => {
    const c = el.closest('input, textarea, select');
    if (c) return c;
    const l = el.closest('label');
    return l && l.control ? l.control : null;
  };
  const choose = (s, want) => {
    const opts = Array.from(s.options), w = norm(want);
    const o = opts.find(o => norm(o.text) === w || norm(o.value) === w)
      || opts.find(o => w && (norm(o.text).startsWith(w) || w.startsWith(norm(o.text)) && norm(o.text)));
    if (!o) return false;
    s.value = o.value; return true;
  };
  // SVG icons have no click(); a dispatched click reaches the same document handlers.
  const press = (el, a) => typeof el.click === 'function' ? el.click()
    : el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, clientX: a.x, clientY: a.y, view: window}));
  const fire = c => { c.dispatchEvent(new Event('input', {bubbles: true})); c.dispatchEvent(new Event('change', {bubbles: true})); };
  const log = [];
  for (const a of job.actions) try {
    scrollTo(0, 0);
    const el = document.elementFromPoint(a.x, a.y);
    if (!el) { log.push({kind: a.kind, hit: null}); continue; }
    const hit = (el.closest('[data-item-index]') || {dataset: {}}).dataset.itemIndex || null;
    const field = (el.closest('[data-field-label]') || {dataset: {}}).dataset.fieldLabel || null;
    if (a.kind === 'click') { press(el, a); log.push({kind: 'click', tag: el.tagName.toLowerCase(), item: hit, field}); continue; }
    const c = control(el);
    if (!c) {   // typing starts with a click, so a type or select on a button or link presses it
      press(el, a); log.push({kind: a.kind, tag: el.tagName.toLowerCase(), item: hit, field, control: null}); continue; }
    c.focus();
    let ok = true;
    if (c.tagName === 'SELECT') ok = choose(c, a.kind === 'select' ? a.option : a.text);
    else if (c.type === 'checkbox' || c.type === 'radio') { c.click(); }
    else { c.value = String(a.kind === 'select' ? a.option : a.text); }
    if (ok) fire(c);
    log.push({kind: a.kind, tag: c.tagName.toLowerCase(), item: hit, field: c.dataset.fieldLabel || field, set: ok});
  } catch (e) { log.push({kind: a.kind, error: String(e)}); }
  scrollTo(0, 0);
  const after = B ? B.getState() : null;
  return JSON.stringify({key: job.key, before, after, log, url: location.href});
})
"""

final class Driver: NSObject, WKNavigationDelegate {
    let window: NSWindow
    let web: WKWebView
    var loaded = false
    var steps: [(@escaping () -> Void) -> Void] = []
    var replayLines: [String] = []

    override init() {
        web = WKWebView(frame: NSRect(origin: .zero, size: viewport), configuration: WKWebViewConfiguration())
        window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: viewport.width, height: viewport.height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
        super.init()
        window.contentView = web
        web.navigationDelegate = self
        web.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent().deletingLastPathComponent())
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        decisionHandler(loaded ? .cancel : .allow)   // the first load only; links and form posts stay put
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if loaded { return }
        loaded = true
        for r in renders { steps.append { done in self.render(r, done) } }
        for r in replays { steps.append { done in self.replay(r, done) } }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { self.next() }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { if !loaded { fail("\(error)") } }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        if !loaded { fail("\(error)") }
    }

    func fail(_ msg: String) {
        FileHandle.standardError.write("webrun: \(page.lastPathComponent): \(msg)\n".data(using: .utf8)!)
        exit(1)
    }

    func next() {
        if steps.isEmpty { finish(); return }
        let step = steps.removeFirst()
        step { DispatchQueue.main.async { self.next() } }
    }

    func finish() {
        if let out = replayOut {
            let text = replayLines.map { $0 + "\n" }.joined()
            do { try text.write(toFile: out, atomically: true, encoding: .utf8) } catch { fail("write: \(error)") }
        }
        exit(0)
    }

    func render(_ r: [String: Any], _ done: @escaping () -> Void) {
        let mode = r["mode"] as? String ?? "fill", out = r["out"] as! String
        web.evaluateJavaScript("\(reset)(\(js(mode)))") { _, error in
            if let error = error { return self.fail("reset: \(error)") }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                self.web.evaluateJavaScript(collect) { result, error in
                    guard let text = result as? String, let data = text.data(using: .utf8),
                          var info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                        return self.fail("collect: \(String(describing: error))")
                    }
                    info["width"] = Int(viewport.width)
                    info["height"] = Int(viewport.height)
                    info["scale"] = 1
                    info["mode"] = mode
                    self.snapshot(out) {
                        do {
                            let json = try JSONSerialization.data(withJSONObject: info, options: [.sortedKeys])
                            try json.write(to: URL(fileURLWithPath: out + ".elements.json"))
                        } catch { self.fail("write: \(error)") }
                        done()
                    }
                }
            }
        }
    }

    func snapshot(_ out: String, _ done: @escaping () -> Void) {
        let conf = WKSnapshotConfiguration()
        conf.rect = NSRect(origin: .zero, size: viewport)
        conf.snapshotWidth = NSNumber(value: Double(viewport.width) / Double(window.backingScaleFactor))
        web.takeSnapshot(with: conf) { image, error in
            guard let image = image else { return self.fail("snapshot: \(String(describing: error))") }
            let w = Int(viewport.width), h = Int(viewport.height)
            let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h, bitsPerSample: 8,
                                       samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                       colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
            rep.size = NSSize(width: w, height: h)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            image.draw(in: NSRect(x: 0, y: 0, width: w, height: h))
            NSGraphicsContext.restoreGraphicsState()
            do { try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out + ".png")) }
            catch { self.fail("write: \(error)") }
            done()
        }
    }

    func replay(_ r: [String: Any], _ done: @escaping () -> Void) {
        web.evaluateJavaScript("\(replayJS)(\(js(r)))") { result, error in
            if let text = result as? String {
                self.replayLines.append(text)
            } else {
                let key = r["key"] as? String ?? "?"
                let line = try! JSONSerialization.data(withJSONObject: ["key": key, "error": "\(String(describing: error))"])
                self.replayLines.append(String(data: line, encoding: .utf8)!)
            }
            done()
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let driver = Driver()
DispatchQueue.main.asyncAfter(deadline: .now() + 300) { driver.fail("timed out") }
app.run()
