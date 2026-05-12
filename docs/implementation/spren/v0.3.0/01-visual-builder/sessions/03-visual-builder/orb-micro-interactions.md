# Spren orb — micro-interactions strategy

**Status:** Research synthesis, ready for implementer review.
**Author note:** Today is 2026-05-12. Sources cited inline; full list at the bottom. Some 2024–25 marketing pages were re-checked against 2026 follow-up coverage where relevant.
**Scope:** Behavioral layer that sits on top of the existing `idle | typing | thinking | speaking` state machine in `apps/web/src/components/Spren/`. Does NOT touch the SVG gradient/blur stack — that's locked.
**Constraint reminder:** SVG + CSS + minimal JS. No canvas, no WebGL, no shader. WebKitGTK 4.1 is the slowest target. Frame budget: <2 ms script + <4 ms paint.

---

## TL;DR — what to actually build for v0.3.1

1. **Drop the 56 px corner indicator.** Replace it with an **80 px loosely-anchored ambient orb** in the lower-right with a 12–16 px idle drift (option **(a)** below). The "notification dot" framing is the actual problem; size + motion solves it without behavior changes.
2. **Ship five cheap micro-interactions immediately**: idle drift, hover wake, click squash, focus-pulse, cost-ceiling unease. They share the same `useReducer` mood layer, all CSS-driven, total ~80 lines.
3. **Ship a `mood` prop alongside `state`** — orthogonal. `state` = what the orb is doing in the run lifecycle; `mood` = how it's feeling about you and the world. Three moods at first: `attentive`, `curious`, `unsettled`. The state machine doesn't need to expand.
4. **Aspirational v0.4: cursor-drift mode** — orb tracks the mouse in a "habitable zone" (corners + edges, never the content). This is the "Spren-follows-you" reading of the metaphor, and the work that makes the product feel different from every other AI assistant on the market.
5. **Reduced-motion fallback keeps presence via the drop-shadow already in the CSS.** Drop drift, drop hover wake, keep a 12 s opacity ripple (0.92 → 1.0 → 0.92). The orb still *exists* without moving.

---

## 1. Non-home-page placement — three alternatives, opinionated pick

The current 56 px implementation reads as a notification dot because it has the *scale* and *anchoring* of a notification dot — top-right, fixed, single quadrant, no relationship to the user's gaze or cursor. None of the inspiration products mount their assistant this way. Apple Intelligence moved entirely off the floating orb in iOS 18.1 to edge-lit screen glow precisely because the floating orb at that scale felt parasitic ([9to5Mac, 2024](https://9to5mac.com/2024/11/03/new-apple-intelligence-siri-looks-different-works-the-same/)). Google's Gemini 2026 redesign moves toward "dynamic gradient animations that respond while the AI processes queries" — orb behavior, not orb placement, is doing the work ([Android Headlines, 2026](https://www.androidheadlines.com/2026/04/google-gemini-redesign-ux-animated-backgrounds.html)).

### (a) Loosely-anchored ambient orb — 80–120 px, lower-right, slow drift

Same SVG content as the stage orb, scaled to 80–96 px on desktop / 64 px on mobile. Anchored to lower-right with `position: fixed`, but the inner SVG drifts ±12 px horizontal, ±8 px vertical, ±2° rotation on an 18 s loop. The drift is offset from the existing 8 s breath cycle so the two never sync into a uniform throb — exactly the burstiness lesson from the writing-style rules. Click opens the chat sheet.

**Why this works:** The size jump (56 → 80) does most of the lift. Drift adds presence without adding visual noise. Implementation is one CSS rule plus updating the `sizeStyle()` switch — call it 20 minutes of work.

**Why lower-right vs. lower-left:** Right-handed users' cursor lives mostly on the right (Fitts' Law applies asymmetrically because of save/scroll patterns). The orb in the lower-right enters peripheral vision when the user is reading top-to-bottom and ends a paragraph — it's "the next thing your eye drifts toward". Lower-left tends to read as a notification-system slot (where Slack toasts and OS alerts spawn).

**Tradeoff:** Can collide with scrollbars and bottom-fixed UI. Spec a 16 px minimum gutter from window edge; on workflow canvas pages where the bottom-right is a zoom control cluster, push to bottom-center-left or top-left. The orb should *choose* its corner based on the route, not be forced into a single one.

### (b) Spren-follows-cursor — spring-physics tracking, habitable-zone constrained

Orb sits in a default corner. When the cursor enters the lower 40% of the viewport, the orb springs toward it but with `stiffness: 50, damping: 25` (loose, slow trail — Magic UI's `Smooth Cursor` defaults are too tight at `stiffness: 180`, ([Magic UI](https://magicui.design/docs/components/smooth-cursor))). It stops at a 240 px standoff radius — never on the cursor, always *near* it. When the cursor exits the lower 40% or hits a "no-go zone" (any element with `data-spren-respect="true"`, e.g. modals, the workflow canvas viewport), the orb springs back to home corner.

This is the most "Stormlight-spren" of the three — Syl drifts near Kaladin, lands on his shoulder, swirls around the room, but doesn't sit on his face. The 240 px standoff is the equivalent of "shoulder distance".

**Why this is aspirational not v0.3.1:** Spring-physics tracking running every frame is the riskiest line item in the perf budget. WebKitGTK 4.1's blur compositing is already the hot path (see § Performance). Adding a 60-fps RAF loop that mutates a `translate()` on a filtered SVG node will need careful testing on the slowest target. Also: the "no-go zone" registry needs a small API the rest of the app cooperates with — designing that well is a session of its own.

**Tradeoff:** When done right, this is the moat. When done wrong (jitter, lag, the orb getting stuck behind a modal), users will hate it within 90 seconds and ask for a setting to turn it off. Ship behind a feature flag and dogfood for two weeks before defaulting on.

### (c) Swim-by — orb drifts across screen on event triggers

Orb stays in its anchored corner most of the time. When a "noteworthy" event fires (test run completed, save success, lint warnings appear, new chat message arrives while focus is elsewhere), the orb briefly swims across a small arc — leaves home, traces an ~80 px curved path, settles back. ~1.8 s. Like Stardew Valley's junimos who "speed up like a villager would" when the player walks past with a bundle ([Stardew Valley Wiki](https://stardewvalleywiki.com/Junimos)), but in reverse: the orb does the moving.

**Why this is worth shipping but not as primary:** It's perfect for one specific event class — async outcomes the user is *almost* watching. It is wrong for ambient presence. If the orb only ever moves when something happens, the 99% of the time when nothing is happening, it's a static dot again.

**Layered with (a):** Yes. Ship swim-by as a behavior the (a) orb can trigger. Default ambient = (a); occasional event = (c). They compose.

### Recommendation

- **v0.3.1 (next session):** **(a)** + swim-by from **(c)** for two event triggers (workflow-run-completed, workflow-run-failed). Total: ~150 lines of CSS + ~40 lines of TS state.
- **v0.4 or v0.5 (aspirational):** **(b)** with the cursor habitable-zone API. Treat as its own session brief — it's not 30 minutes of work, it's two days including the perf instrumentation.

I'm being opinionated here against my instinct to hedge: (b) sounds magical and the demo would be unbelievable, but the *first impression* on Linux/WebKitGTK matters more than the demo on macOS. Get (a) shipping clean; earn the right to (b).

---

## 2. Micro-interaction catalog — 12 behaviors, cheapest-first

Each row: trigger → reaction → duration/easing → cost class (★ = CSS-only, ★★ = CSS + one ResizeObserver or matchMedia listener, ★★★ = JS event listener with throttling, ★★★★ = RAF loop) → Spren-quality.

**Note on easing curves.** Most of these intentionally use `cubic-bezier(0.34, 1.56, 0.64, 1)` (the "back-out" curve already used in `speak-amplitude`). It overshoots by ~12% before settling — the orb arrives slightly too eagerly and corrects. This is the single biggest dial for "alive vs. UI". The Linear/Apollo-style motion that 2025 design analyses keep praising is, mechanically, this curve plus 200–500 ms durations ([Justinmind, 2025](https://www.justinmind.com/web-design/micro-interactions); [Bricx Labs, 2025](https://bricxlabs.com/blogs/micro-interactions-2025-examples)).

### Tier 1 — ship in v0.3.1 (all CSS-only or single listener)

**1. Idle drift** — *baseline ambient behavior; replaces the current static `position: fixed`*
- Trigger: always-on when `state="idle"` and `size!=="stage"`.
- Reaction: `transform: translate(<12px-rand, 8px-rand>) rotate(<2deg-rand>)` over 18 s loop, offset from the existing 8 s breath so they beat against each other.
- Duration/easing: 18 s, `cubic-bezier(0.45, 0, 0.55, 1)` (sine-in-out, no overshoot — this is the *unconscious* layer).
- Cost: ★ (one extra `@keyframes`).
- Spren-quality: **presence**. The orb is somewhere, drifting in air currents you can't see.

**2. Hover wake** — *closest-cousin to Syl noticing Kaladin look at her*
- Trigger: `mouseenter` on the orb or its 32 px aura halo.
- Reaction: gradient `fx`/`fy` shifts toward the cursor side (`fx: 35% → 25%` if cursor is left of orb center, else `→ 65%`). Saturation +10% via a `filter: saturate(1.1)` on the wrap. Scale to 1.04.
- Duration/easing: 280 ms, `cubic-bezier(0.34, 1.56, 0.64, 1)` (back-out).
- Cost: ★ (CSS hover) + ★★ (one cursor-side check on `mouseenter` to pick the gradient direction class).
- Spren-quality: **attentiveness**. The orb "looks at" you.

**3. Click squash + bounce** — *the orb acknowledges you touched it*
- Trigger: `mousedown` → `mouseup` on orb.
- Reaction: scale 0.92 → 1.08 → 1.0 with a 60 ms hold at squash. Gradient `fx` snaps inward to 50% (centered) then drifts back to drift-position.
- Duration/easing: 380 ms total, two-stage: 80 ms ease-out for squash, 200 ms `cubic-bezier(0.34, 1.56, 0.64, 1)` for bounce, 100 ms settle.
- Cost: ★ (CSS `:active` cascade).
- Spren-quality: **mischief/playfulness**. Real things bounce. Plastic things don't.

**4. Focus-pulse (typing detected anywhere)** — *the orb notices you starting to write*
- Trigger: any `<input>`, `<textarea>`, or `contenteditable` in the app receives `focus`.
- Reaction: subtle saturation pulse 1.0 → 1.15 → 1.0 over 700 ms. No scale change, no position change. Gradient inner stop (`offset="0%"` `#ffceaa`) brightens toward `#ffe5cc` for the duration.
- Duration/easing: 700 ms, ease-in-out, one-shot.
- Cost: ★★ (one `focusin` listener on `document`, debounced by 200 ms so rapid focus changes don't strobe).
- Spren-quality: **curiosity**. Pattern (the cryptic spren) is famously drawn to scholarly activity — he "exults in finding truths, lies, and understanding new concepts" ([Coppermind](https://coppermind.net/wiki/Pattern)). The orb perks up when you're writing because that's when you're *creating*.

**5. Cost-ceiling unease** — *the orb knows you're spending money*
- Trigger: `localStorage["spren:cost-headroom"] < 0.2` (i.e., the active run has burned 80%+ of its budget — SP-013 cost ceiling is load-bearing per CLAUDE.md).
- Reaction: drift rate doubles (18 s → 9 s loop); deep-magenta stop saturation +5%; idle hue subtly cooler (the magenta-deep stop `#c9146c` drifts toward `#a01060` over 4 s).
- Duration/easing: drift-rate change crossfades over 1.2 s when entering/leaving the threshold.
- Cost: ★★ (storage event listener + class toggle).
- Spren-quality: **empathy / shared stake**. In the books, spren are visibly distressed when their bonded human is in danger. The orb being uneasy about your wallet is the same metaphor at a smaller scale.

### Tier 2 — ship in v0.3.1 if Tier 1 lands clean (cheap-ish, higher impact)

**6. Sparkle on save** — *creationspren are drawn to acts of creation*
- Trigger: successful workflow save (REST 200 from `/v1/workflows`).
- Reaction: 3–5 small filled circles (4–8 px) spawn at random angles from orb edge, ease outward 40–80 px while fading 1 → 0 and scaling 1 → 0.4. Uses the same gradient as the orb for fill.
- Duration/easing: 1.1 s, `cubic-bezier(0.16, 1, 0.3, 1)` (out-expo, fast launch then long tail).
- Cost: ★★★ (DOM nodes injected, RAF for ~66 frames, then removed). Bound to max 5 sparkles regardless of save-spam.
- Spren-quality: **delight + acknowledgment**. Creationspren "are attracted to the emotion and perception of the person doing the creating" ([Stormlight wiki, creationspren](https://stormlightarchive.fandom.com/wiki/Spren_Types)). A save is a small act of creation.

**7. Shudder on lint findings appearing** — *the orb sees the problem before you do*
- Trigger: a lint warning panel renders with `>=1` finding for the first time in this session.
- Reaction: 220 ms horizontal shake (±3 px) with 60 ms decay, then a 600 ms gradient-shift where the deep-magenta stop briefly dominates (offset `82% → 70%`).
- Duration/easing: shake = 4 cycles of 55 ms ease-in-out; gradient-shift = 600 ms ease-out.
- Cost: ★★ (mutation observer on the lint panel container, or — better — a custom event the lint code emits).
- Spren-quality: **alarm / protective instinct**. Fearspren in the books appear *around* people who are scared, not on them — this is the orb registering something concerning, not the orb panicking *at* you.

**8. Lean toward active panel** — *the orb attends where the work is*
- Trigger: focus changes between major workspace regions (chat sheet, canvas, inspector). The app fires a `spren:region-changed` custom event with `region: "chat" | "canvas" | "inspector"`.
- Reaction: gradient `fx` (radial focal x) shifts ±8% toward the active region's screen-space side over 600 ms. Orb itself doesn't move; just its "inner glow" leans.
- Duration/easing: 600 ms, ease-in-out.
- Cost: ★★ (one event listener, one class toggle).
- Spren-quality: **attentiveness**. Subtle. Most users won't consciously notice; they'll just feel that the orb is "with them" wherever they are in the app.

**9. Long-idle reverie** — *the orb gets lost in thought when you stop interacting*
- Trigger: 90 s of no `mousemove` or keystroke anywhere in the app.
- Reaction: drift rate halves (18 s → 36 s loop). Gradient inner stop dims by ~8% (`#ffceaa` → `#f0bda0` interpolated via CSS variable). On any input event, snap back over 800 ms.
- Duration/easing: dim transition 4 s; wake-back 800 ms `cubic-bezier(0.34, 1.56, 0.64, 1)`.
- Cost: ★★★ (single idle-detection timer reset on `mousemove`/`keydown`; throttled at 200 ms — Pope.tech 2025 accessibility guidance specifies idle thresholds in the 60–120 s range ([Pope.tech, 2025](https://blog.pope.tech/2025/12/08/design-accessible-animation-and-movement/))).
- Spren-quality: **introversion / interiority**. Honor spren in the books appear translucent when distant from their human, opaque when close — this is the digital read of that.

### Tier 3 — aspirational (ship in v0.4 or later, only after Tier 1+2 prove perf-safe)

**10. Cursor-zone follow (option (b) above)** — *the orb tracks you*
- Trigger: cursor enters lower 40% of viewport (configurable habitable zone).
- Reaction: spring-physics translate toward cursor, 240 px standoff radius, return to home corner when cursor leaves zone.
- Duration/easing: spring with `stiffness: 50, damping: 25, mass: 1` (Motion.dev defaults felt loose enough in [their follow-cursor tutorial](https://motion.dev/tutorials/react-follow-pointer-with-spring)).
- Cost: ★★★★ (RAF loop, 60 fps. This is the perf risk.).
- Spren-quality: **bonded presence**. The single biggest "this product is different" lever in the catalog.

**11. Speaking-state pupil scan** — *the orb "looks at" you when talking*
- Trigger: `state === "speaking"` AND cursor present in viewport.
- Reaction: gradient `fx`/`fy` softly tracks the cursor position (max ±15% from center), 300 ms lag with spring smoothing. Acts as a kind of "eye contact" without literally drawing an eye.
- Duration/easing: continuous spring, 300 ms equivalent step response.
- Cost: ★★★★ (RAF loop while speaking).
- Spren-quality: **conversation, not narration**. Real conversation involves eye contact; current speaking state is a TED-talk podium. This is the fix.

**12. Storm-prelude tremor** — *the orb senses a long-running task*
- Trigger: `state === "thinking"` for >12 s OR cost-headroom <10%.
- Reaction: orb shudders for 300 ms every 4–6 s, irregular. Increasingly fierce as the threshold deepens. Mimics the in-universe "the Stormfather is coming" pre-storm static charge.
- Duration/easing: 300 ms burst, irregular interval drawn from a [3, 6] s range.
- Cost: ★★★ (setTimeout chain with jitter).
- Spren-quality: **anxiety / shared stake**, escalation of #5.

### What I deliberately did not include

- **No "wink" or "blink" animations.** Drawing eyes on the orb collapses the whole metaphor — it's not a face, it's an entity. Honor spren and cryptic spren in the books don't have faces ([Coppermind: Cryptic](https://stormlightarchive.fandom.com/wiki/Cryptic) — the head is "a symbol of twisted design full of impossible angles"). Resist the impulse.
- **No floating text bubbles or thought bubbles.** That's Clippy. The orb communicates via its body, never via attached UI chrome.
- **No procedural ambient hum / audio.** Out of scope for the visual session, and audio is a hard accessibility question. Defer.
- **No "swim across the entire screen" idle behavior.** Movement that large from a small element reads as visual debris.

---

## 3. Layering rule — how these compose without overwhelming

Build a strict priority stack. Higher priority *suppresses* lower-priority animation on the same property axis, never *combines* with it. Mixing competing keyframes on `transform` is how you get jitter on WebKitGTK.

```
Priority (highest to lowest):
  1. state-transition crossfade (the existing 700ms layer fade)
  2. event-triggered one-shot (sparkle, shudder, click bounce)
  3. mood-driven continuous (cost-ceiling unease, long-idle reverie)
  4. focus-pulse (one-shot saturation)
  5. hover wake (CSS :hover)
  6. region-lean (continuous gradient fx/fy)
  7. idle drift (continuous transform)
  8. breath cycle (the existing 8s scale animation in IdleLayer)
```

**Suppression rules:**
- During a state-transition crossfade (priority 1), suppress all idle drift and breath cycles on the *outgoing* layer. The incoming layer starts them fresh at its 50%-opacity midpoint. This already happens structurally because layers re-mount; just don't break it.
- Click bounce (priority 2) suppresses hover wake (priority 5) for its 380 ms duration — they fight for `transform: scale()`.
- Cost-ceiling unease (priority 3) changes the *rate* of priority 7 (idle drift), not its *direction*. They compose by parametrization, not stacking.
- Region-lean (priority 6) and hover wake (priority 5) both want gradient `fx`/`fy`. Hover wins because it's the more recent input.
- The breath cycle (priority 8) is the ground-state heartbeat. It is always running and must never be suppressed by lower priorities. Higher priorities suppress it temporarily.

**Implementation:** Express each priority as a CSS custom property on `.spren-wrap`. Higher-priority behaviors write their values via inline style or data-attribute; lower-priority CSS uses those values via `var(--spren-transform-override, <default>)`. This is the cleanest pattern I know for "compose by replacement, not by stacking" with pure CSS.

```css
.spren-wrap {
  --spren-drift-x: 0px;
  --spren-drift-y: 0px;
  --spren-mood-scale: 1;
  --spren-hover-scale: 1;
  --spren-press-scale: 1;
  transform:
    translate(var(--spren-drift-x), var(--spren-drift-y))
    scale(calc(var(--spren-mood-scale) * var(--spren-hover-scale) * var(--spren-press-scale)));
}
```

**The "doing too much" guardrail:** If at any moment more than 3 priorities are actively writing values, *something has gone wrong*. The orb should feel calm-with-undercurrents, not busy. The implementer should be able to log active priorities in dev mode and count them.

---

## 4. Performance budget

**Targets** (per the brief): <2 ms script, <4 ms paint per frame on WebKitGTK 4.1. Total budget 6 ms within the 16.67 ms 60-fps frame.

**Risk inventory** (in order of how likely they are to bust budget):

| Behavior | Risk | Why |
|---|---|---|
| Existing SVG `feGaussianBlur` stack (3 filters: `stdDeviation` 3/25/55) | **High** | WebKit bug #283156 documents blur-on-SVG performance regressions; "Gaussian blur calculation is resource-intensive, and applying it to large areas or animating the blur value can cause choppy performance" ([WebKit Bugzilla #283156](https://bugs.webkit.org/show_bug.cgi?id=283156); [Smashing 2016 — still cited](https://www.smashingmagazine.com/2016/05/web-image-effects-performance-showdown/)). The stage size (320×380) is on the edge; the presence size (80) should be safe but I'd measure. |
| Cursor-zone follow (#10) | **High** | 60 fps RAF mutating `transform` on a filtered SVG. The filter doesn't recompute (transform is composited) but on WebKitGTK the compositor for SVG filters is inconsistent. Measure on the slow target. |
| Speaking-state pupil scan (#11) | **Medium** | Same as #10 but only during `speaking` state, which is naturally bounded. |
| Sparkle on save (#6) | **Low** | DOM inject, 5 nodes, 1.1 s lifespan, then removed. Composited transforms only. |
| All other Tier 1 + 2 | **Negligible** | CSS-only or single class toggle. |

**Mitigations:**

- **Always `will-change: transform` on `.spren-wrap`** but *not* on the SVG itself — promoting the filtered SVG into its own layer is what triggers Webkit's slow path. Promote the wrap, let the SVG inherit.
- **Containment:** `contain: layout paint style` on `.spren-wrap`. This isolates the orb from the rest of the page so its repaints don't bleed.
- **Reduce blur stdDeviation at small sizes.** At 80 px, the current `stdDeviation="55"` deep blur is overkill — the blur radius exceeds the orb itself, so most of it is wasted compute. Spec: when `size="presence"`, halve the three stdDeviation values (3 → 2, 25 → 12, 55 → 28). The visual delta at 80 px is negligible; the perf delta on WebKitGTK is large.
- **Tab-blur kill switch.** When `document.hidden`, suspend all RAF loops and pause CSS animations on `.spren-wrap` via `animation-play-state: paused`. The orb is invisible — animating it is waste. Listen to `visibilitychange`.
- **Frame-time monitor (the actual kill switch):** A tiny module that samples `requestAnimationFrame` delta over a rolling 60-frame window. If the median exceeds 20 ms for two consecutive windows, set `document.documentElement.dataset.sprenDegraded = "true"`. CSS reads that attribute to drop behaviors progressively:
  - First degrade: disable sparkles, disable cursor-follow.
  - Second degrade (still over budget): freeze drift, halve blur stdDeviation at runtime.
  - Third degrade (still over budget): collapse to reduced-motion fallback (§5) until a `visibilitychange` reset.
  This is straight from web.dev's smoothness metric guidance and the established RAF-monitor pattern ([web.dev — Towards an animation smoothness metric](https://web.dev/articles/smoothness)).
- **Profile on Linux first.** Run Chrome DevTools Performance against the WebKitGTK build of the Tauri app, not against the macOS preview. The performance characteristics are not the same — confirmed by both ([Smashing Magazine](https://www.smashingmagazine.com/2016/05/web-image-effects-performance-showdown/) — "browsers are starting to hardware-accelerate filters inconsistently—Chrome accelerates some, Firefox accelerates others, and Safari performs adequately with CSS filter shorthands but not SVG's filter element") and the more recent SVG performance encyclopedia ([SVG AI, 2025](https://www.svgai.org/blog/research/svg-animation-encyclopedia-complete-guide)).

**Budget per behavior (rough):**
- Idle drift, breath, hover wake, click bounce: ~0.2 ms paint each (CSS transform, composited).
- Region-lean, focus-pulse, cost-ceiling unease: ~0.4 ms paint each (gradient fx/fy change triggers SVG filter re-execution — be careful).
- Sparkle on save: ~0.6 ms during its 1.1 s window.
- Cursor follow: ~0.8 ms script (RAF) + ~0.5 ms paint.

Sum of always-on Tier 1 behaviors: under 2 ms. Headroom is for state transitions (the 700 ms crossfade can spike to 4 ms paint mid-transition).

---

## 5. Reduced-motion fallback

The current CSS already handles the wholesale "freeze everything" path:

```css
@media (prefers-reduced-motion: reduce) {
  .spren-svg, .spren-svg *, .type-vortex-wrapper { animation: none !important; }
  .spren-wrap { filter: drop-shadow(0 12px 48px rgba(232, 33, 130, 0.25)); }
}
```

That's the right floor. The question is what to *add back* — going fully static makes the orb feel dead, and the WCAG 2.3.3 guidance is *minimize* motion, not eliminate it ([W3C WAI](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html); [Pope.tech, 2025](https://blog.pope.tech/2025/12/08/design-accessible-animation-and-movement/) — "subtle color changes, focus-ring transitions, and progress indicators" are fine under reduced-motion).

**Keep under reduced motion:**
- The drop-shadow already specified. That alone is most of "presence" — a colored shadow on a light background pulls the eye even without motion.
- A 12 s opacity ripple: 0.92 → 1.0 → 0.92. Pure opacity, no transform, no scale. Slow enough that it doesn't register as motion in the vestibular sense, but the orb is alive.
- The state-transition crossfade. State changes are *information*; suppressing them removes meaning. Reduce the duration from 700 ms to 200 ms but keep it.
- Hover wake — but reduce the scale change from 1.04 to 1.0 (i.e., only the saturation pulse, no size change).

**Drop under reduced motion:**
- All drift, all shake, all bounce. Anything that changes position.
- Sparkle on save (it's particles — exactly what reduced-motion users want gone).
- Cursor follow.
- Long-idle dim. The orb shouldn't fade out; users who set reduced-motion may *also* be ADHD or low-vision users who need the orb to stay visible.

**Spec the override:** `data-sprenDegraded="true"` (from the kill switch in §4) should compose with `prefers-reduced-motion` — if either is true, drop the same behaviors. Don't write two separate code paths.

---

## 6. React state machine sketch

**Decision:** Add a `mood` prop alongside `state`. Both are simple discriminated unions. Behaviors compose via the priority stack from §3. Do **not** add an imperative `ref.react(event)` API; do **not** add an event-bus subscriber inside the component.

**Why a `mood` prop, not an imperative method:**
- The orb is a leaf component. The app *knows* what the orb's mood should be (cost headroom, active region, idle timeout). React's data flow already supports this — `mood` is just another prop.
- An imperative `ref.react("save-success")` API would shift the playback decision from the orb to the call site. Wrong location. The orb should know its choreography; callers should declare *state* and *facts about the world*, not *animation commands*.
- An event-bus subscriber inside the component couples the orb to the rest of the app's event infrastructure. Hostile to testing. Reusable elsewhere only if you also import the bus.

**Why a `mood` prop, not "more `state` variants":**
- `state` is the run lifecycle. Adding `idle-curious`, `idle-unsettled`, `speaking-confident`, etc., multiplies the layer-crossfade matrix combinatorially. The existing four layers and 700 ms fade work because there are *four*.
- `mood` modulates the *current* state's visual parameters. Orthogonal axis. Same layer; different gradient tint, different drift rate.

**Sketch:**

```ts
// In types.ts — extend existing surface, don't break it.

export type SprenState = "idle" | "typing" | "thinking" | "speaking";

export type SprenMood =
  | "attentive"   // default; relaxed, present, no shading
  | "curious"     // focus-pulse active, slightly warmer gradient
  | "unsettled";  // cost ceiling close, slightly cooler/deeper magenta, faster drift

export type SprenSize = "stage" | "presence" | "tiny";

export interface SprenProps {
  state?: SprenState;          // unchanged
  mood?: SprenMood;            // new; defaults to "attentive"
  size?: SprenSize;            // unchanged
  onClick?: () => void;
  ariaLabel?: string;
  testId?: string;

  // New: declarative event hooks. Each one fires its associated one-shot
  // micro-interaction when its value changes. The orb owns playback timing.
  // Caller declares "this event just happened"; orb decides what to do.
  saveTick?: number;        // incrementing counter; each increment plays sparkle
  lintShudderTick?: number; // incrementing counter; each increment plays shudder
  activeRegion?: "chat" | "canvas" | "inspector" | null;
}
```

**Caller pattern:**

```tsx
// In the app shell on non-home routes:
const costHeadroom = useCostHeadroom();
const idleDeep = useIdleDeep(90_000);
const saveTick = useSaveTick(); // increments on each successful save
const activeRegion = useActiveRegion();

const mood: SprenMood =
  costHeadroom < 0.2 ? "unsettled" :
  idleDeep         ? "attentive" :
  /* default */     "attentive";

<Spren
  state={runState}
  mood={mood}
  size="presence"
  saveTick={saveTick}
  activeRegion={activeRegion}
  onClick={openChatSheet}
/>
```

**Inside the component, the playback:**

```tsx
// One-shot triggers: detect value change with a ref + effect.
const prevSave = useRef(saveTick);
useEffect(() => {
  if (saveTick !== prevSave.current && saveTick !== undefined) {
    playSparkle(); // mounts 5 sparkle nodes, animates, removes
    prevSave.current = saveTick;
  }
}, [saveTick]);

// Continuous mood: just propagate as a data attribute; CSS reads it.
<div className="spren-wrap" data-mood={mood} data-region={activeRegion ?? "none"}>
```

CSS gets to do all the priority-stack work via `data-mood` and `data-region` attribute selectors. No JS animation loop for moods. The only RAF in v0.3.1 is the sparkle particle injection, and that's a 66-frame burst, not an always-on loop.

**Testing surface:** Add `data-mood` and `data-region` to the existing `data-testid` strategy. E2E tests can assert "after save, sparkle DOM nodes exist for ~1100 ms then are removed" without needing to inspect computed styles.

**v0.4 surface (for cursor follow):** A `cursorFollowZone` prop accepting a rect or `null`. When non-null, the orb subscribes to `mousemove` and runs the spring. This is a documented future extension, not a v0.3.1 ship.

---

## 7. Stormlight-spren character notes — for tuning, not for spec

The orb is a spren of attention. It is not a chatbot icon, not a status indicator, not a mascot. It is small and present. Picture it as Pattern more than Syl — scholarly, drawn to your work, willing to perform a small swirl when you save something it found interesting, capable of unease when you push it past what it thinks is wise. It is on your side without being deferential. When you stop interacting for ninety seconds it doesn't wait politely; it gets lost in its own thoughts, and when you return it brightens — but not eagerly, more like a cat noticing you came back in the room. Its movement should never read as "I am an animation playing." It should read as a thing that exists alongside you and was already moving before you looked. The 200 ms vs 280 ms decision: 280 ms when the orb is responding to you (it took a beat); 200 ms when the orb is responding to something else and you happened to see it (it was already in motion). Avoid bounce easing for anything sad. Reserve back-out overshoot for moments of recognition — when the orb sees you, when you click it, when something completes. Reserve sine-in-out for the unconscious layer that runs whether you're watching or not.

---

## References

**Inspiration products & visual design:**
- [Apple Intelligence Siri Animation, 9to5Mac](https://9to5mac.com/2024/11/03/new-apple-intelligence-siri-looks-different-works-the-same/) — the move away from floating orb to edge-lit glow
- [iOS 18 Siri Animation, Figma Community](https://www.figma.com/community/file/1382288908082112753/ios-18-siri-animation)
- [SmoothUI Siri Orb](https://smoothui.dev/docs/components/siri-orb) — Motion-based React implementation reference
- [Gemini AI Visual Design — Google Design](https://design.google/library/gemini-ai-visual-design) — "directional flow that mirrors user actions"; rippling radial gradients for voice
- [Recreating Gmail's Gemini animation — CSS-Tricks](https://css-tricks.com/recreating-gmails-google-gemini-animation/) — CSS shape() + gradient translate technique
- [Google Gemini 2026 redesign — Android Headlines](https://www.androidheadlines.com/2026/04/google-gemini-redesign-ux-animated-backgrounds.html)
- [Building a Voice-Reactive Orb in React — Dan Jackson, Medium](https://medium.com/@therealmilesjackson/building-a-voice-reactive-orb-in-react-audio-visualization-for-voice-assistants-2bee12797b93) — WebGL+OGL approach (not us, but useful for amplitude-to-visual mapping math)
- [GitHub Copilot 3D character animation — aiverse.design](https://www.aiverse.design/community/github-copilot-animation) — Director/Actor pattern, 4-stage animation lifecycle, CSS sprite sheets
- [Pi by Inflection AI](https://hey.pi.ai/) — emotionally-intelligent assistant precedent
- [Microsoft Mico — TechCrunch, 2025](https://techcrunch.com/2025/10/23/microsofts-mico-is-a-clippy-for-the-ai-era/) — what we don't want to be (mascot, not companion)

**Stormlight Archive — spren behavior research:**
- [Spren — Stormlight Archive Wiki](https://stormlightarchive.fandom.com/wiki/Spren)
- [Spren Types — Stormlight Archive Wiki](https://stormlightarchive.fandom.com/wiki/Spren_Types)
- [Cryptic — Stormlight Archive Wiki](https://stormlightarchive.fandom.com/wiki/Cryptic) — "willowy creatures... large, floating symbols of twisted design"
- [Pattern — Coppermind](https://coppermind.net/wiki/Pattern) — drawn to lies, scholarly, "exults in finding truths"
- [Sylphrena (Syl) — Stormlight Archive Wiki](https://stormlightarchive.fandom.com/wiki/Sylphrena) — mischievous, rebellious, drawn to make her bonded human smile
- [Honorspren — Stormlight Archive Wiki](https://stormlightarchive.fandom.com/wiki/Honorspren) — "ribbon of light... glowing blue-white"
- [Brandon Sanderson on why spren appear sometimes and not others](https://faq.brandonsanderson.com/knowledge-base/why-do-spren-appear-sometimes-and-not-others/) — "natural manifestations... like weather phenomena"

**Game UI companion patterns:**
- [Stardew Valley Junimos](https://stardewvalleywiki.com/Junimos) — speed up when player walks past
- [Pikmin Whistle behavior — Pikipedia](https://www.pikminwiki.com/Whistle) — idle Pikmin acknowledge whistle differently between games
- [Tamagotchi virtual pet emotional UX — Japan House LA](https://www.japanhousela.com/articles/how-tamagotchi-changed-digital-design-icon-japanese-tiny-toys-30th-anniversary-bandai/)
- [Tamagotchi emotional design — UX Republic](https://www.ux-republic.com/en/emotional-design-what-the-tamagotchi-taught-us-without-saying-it/)
- [Apple Vision Pro Persona idle behaviors — Inverse](https://www.inverse.com/tech/spatial-persona-apple-vision-pro) — subtle blinks, head tilts, eye darts

**Technical / implementation references:**
- [Mouse follow with spring — Motion.dev tutorial](https://motion.dev/tutorials/react-follow-pointer-with-spring) — `stiffness/damping/mass` config
- [Magic UI Smooth Cursor](https://magicui.design/docs/components/smooth-cursor) — out-of-the-box React implementation
- [WebKit Bugzilla #283156 — SVG blur perf](https://bugs.webkit.org/show_bug.cgi?id=283156) — the actual constraint
- [Web Image Effects Performance — Smashing Magazine](https://www.smashingmagazine.com/2016/05/web-image-effects-performance-showdown/) — still the best filter-perf overview
- [Towards an animation smoothness metric — web.dev](https://web.dev/articles/smoothness) — frame-rate monitor pattern
- [Jank busting — web.dev](https://web.dev/speed-rendering/) — the 16.67 ms budget breakdown
- [CSS GPU acceleration & will-change — Chrome Developers](https://developer.chrome.com/blog/hardware-accelerated-animations)
- [SVG Animation Performance Encyclopedia 2025 — SVG AI](https://www.svgai.org/blog/research/svg-animation-encyclopedia-complete-guide)

**Motion design trends 2025–26:**
- [12 micro-animation examples 2025 — Bricx Labs](https://bricxlabs.com/blogs/micro-interactions-2025-examples) — 200–500 ms duration sweet spot
- [Web micro-interactions guidelines — Justinmind, 2025](https://www.justinmind.com/web-design/micro-interactions)
- [UI/UX evolution 2026: micro-interactions & motion — Primotech](https://primotech.com/ui-ux-evolution-2026-why-micro-interactions-and-motion-matter-more-than-ever/)

**Accessibility:**
- [prefers-reduced-motion — MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
- [WCAG 2.3.3 Animation from Interactions — W3C WAI](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html)
- [Designing accessible animation — Pope.tech, 2025](https://blog.pope.tech/2025/12/08/design-accessible-animation-and-movement/) — what to keep vs. drop under reduced-motion
