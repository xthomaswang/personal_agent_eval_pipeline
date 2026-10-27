// Render a page to a screenshot, with the boxes of its elements (the ground truth for the probe).
//
//   swift render.swift <page.html> <out-prefix> [scale]
//
// Loads the page in an off-screen WKWebView with a 1280x1024 CSS-pixel viewport, waits for it to
// settle, and writes:
//   <out-prefix>.png            the viewport, 1280*scale x 1024*scale pixels (scale 1 or 2; WebKit
//                               rasterizes at that scale, so 2 is a Retina capture, not an upscale)
//   <out-prefix>.elements.json  {"scale", "width", "height", "qa": {data-qa id: box},
//                               "qa_labels": {data-qa id: [boxes of its <label>s]}, "ax": [...]}
// Boxes are [x0, y0, x1, y1] in screenshot pixels. "ax" is an accessibility-style list built from the
// DOM, in document order: the visible interactive elements, headings, labels and table cells (role,
// name, placeholder, value, checked, box) and the other visible text (role "text"): what a harness
// reading the accessibility tree would see. It needs no system permission.

import AppKit
import WebKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: swift render.swift <page.html> <out-prefix> [scale]\n".data(using: .utf8)!)
    exit(2)
}
let page = URL(fileURLWithPath: args[1]).standardizedFileURL
let outPrefix = args[2]
let scale = args.count > 3 ? Double(args[3]) ?? 1 : 1
let viewport = NSSize(width: 1280, height: 1024)

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
    if (el.id) { const l = document.querySelector(`label[for="${el.id}"]`); if (l) return text(l); }
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
  const qa = {}, qaLabels = {};
  document.querySelectorAll('[data-qa]').forEach(el => {
    if (!visible(el)) return;
    qa[el.dataset.qa] = box(el);
    // Clicking a control's label acts on the control, so its labels count as part of it.
    const labels = el.labels ? Array.from(el.labels).filter(visible).map(box) : [];
    if (labels.length) qaLabels[el.dataset.qa] = labels;
  });
  const sel = 'input, select, textarea, button, a, [role], label, h1, h2, h3, h4, th, td, img[alt]';
  const ax = [];
  document.querySelectorAll('body *').forEach(el => {
    if (!visible(el) || (el.type || '').toLowerCase() === 'hidden') return;
    if (!el.matches(sel)) {
      // Text outside the listed elements: the static text an accessibility tree also shows.
      const own = Array.from(el.childNodes).filter(n => n.nodeType === Node.TEXT_NODE)
        .map(n => n.textContent).join(' ').replace(/\\s+/g, ' ').trim();
      if (own && !el.closest(sel)) ax.push({role: 'text', name: own.slice(0, 200), box: box(el)});
      return;
    }
    const item = {role: role(el), name: name(el), box: box(el)};
    if (el.dataset.qa) item.qa = el.dataset.qa;
    if (el.placeholder && el.placeholder !== item.name) item.placeholder = el.placeholder;
    if ('value' in el && el.tagName !== 'BUTTON' && el.value && !['checkbox', 'radio'].includes(el.type)) item.value = el.value;
    if (el.tagName === 'SELECT') item.value = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : '';
    if (['checkbox', 'radio'].includes(el.type)) item.checked = el.checked;
    if (el.getAttribute('aria-checked')) item.checked = el.getAttribute('aria-checked') === 'true';
    ax.push(item);
  });
  return JSON.stringify({qa, qa_labels: qaLabels, ax, scrollHeight: document.documentElement.scrollHeight});
})()
"""

final class Renderer: NSObject, WKNavigationDelegate {
    let window: NSWindow
    let web: WKWebView

    override init() {
        web = WKWebView(frame: NSRect(origin: .zero, size: viewport), configuration: WKWebViewConfiguration())
        // Off screen and never ordered front: nothing appears on the user's display.
        window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: viewport.width, height: viewport.height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
        super.init()
        window.contentView = web
        web.navigationDelegate = self
        web.loadFileURL(page, allowingReadAccessTo: page.deletingLastPathComponent())
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { self.capture() }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { fail("\(error)") }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { fail("\(error)") }

    func fail(_ msg: String) {
        FileHandle.standardError.write("render: \(msg)\n".data(using: .utf8)!)
        exit(1)
    }

    func capture() {
        web.evaluateJavaScript(collect) { result, error in
            guard let json = result as? String, let data = json.data(using: .utf8),
                  var info = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return self.fail("collect: \(String(describing: error))")
            }
            if let h = info["scrollHeight"] as? Double, h > viewport.height + 1 {
                FileHandle.standardError.write("render: warning: the page is \(Int(h)) px tall; only the first 1024 are captured\n".data(using: .utf8)!)
            }
            let scaleBox: (Any) -> [Int] = { b in (b as! [Double]).map { Int(($0 * scale).rounded()) } }
            info["qa"] = (info["qa"] as! [String: Any]).mapValues(scaleBox)
            info["qa_labels"] = (info["qa_labels"] as! [String: Any]).mapValues { ($0 as! [Any]).map(scaleBox) }
            info["ax"] = (info["ax"] as! [[String: Any]]).map { item -> [String: Any] in
                var item = item; item["box"] = scaleBox(item["box"]!); return item
            }
            info["scale"] = scale
            info["width"] = Int(viewport.width * scale)
            info["height"] = Int(viewport.height * scale)
            info.removeValue(forKey: "scrollHeight")
            self.snapshot(info)
        }
    }

    func snapshot(_ info: [String: Any]) {
        let conf = WKSnapshotConfiguration()
        conf.rect = NSRect(origin: .zero, size: viewport)
        let backing = window.backingScaleFactor
        conf.snapshotWidth = NSNumber(value: Double(viewport.width) * scale / Double(backing))
        web.takeSnapshot(with: conf) { image, error in
            guard let image = image else { return self.fail("snapshot: \(String(describing: error))") }
            let w = Int(viewport.width * scale), h = Int(viewport.height * scale)
            let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h, bitsPerSample: 8,
                                       samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                       colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
            rep.size = NSSize(width: w, height: h)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            image.draw(in: NSRect(x: 0, y: 0, width: w, height: h))
            NSGraphicsContext.restoreGraphicsState()
            do {
                try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: outPrefix + ".png"))
                let json = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
                try json.write(to: URL(fileURLWithPath: outPrefix + ".elements.json"))
            } catch { self.fail("write: \(error)") }
            exit(0)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let renderer = Renderer()
DispatchQueue.main.asyncAfter(deadline: .now() + 30) { renderer.fail("timed out") }
app.run()
