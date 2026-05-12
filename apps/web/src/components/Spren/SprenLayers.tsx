/**
 * Per-state SVG layers for the Spren orb.
 *
 * Each layer renders one of the four reactive states (idle / typing /
 * thinking / speaking) as a self-contained `<svg>` block. The parent
 * `Spren.tsx` mounts all four layers and toggles their `data-active`
 * attribute via the state machine; the CSS crossfade in `Spren.css`
 * fades the outgoing layer to scale 0.92 + opacity 0 while the incoming
 * layer fades to scale 1 + opacity 1 over 700ms.
 *
 * Shared invariant: every layer's "rest" pose is the same converged
 * gradient egg, so when both layers sit at 50% opacity during a
 * crossfade, the visual sum reads as one calm orb — regardless of where
 * in the outgoing layer's own keyframe timeline the transition began.
 *
 * The gradient + blur stack matches the user-approved reference at
 * docs/implementation/spren/sessions/v0.3.0/03-visual-builder/spren-orb-v4.html,
 * with two refinements:
 *   1. Coral-dominant stops (peach → coral → magenta → magenta-deep),
 *      replacing the v4 reference's soft-pink intermediate.
 *   2. The asymmetric egg path leans subtly right at the top per the
 *      inspiration image (assets/spren-inspiration.png), where v4 used a
 *      mirror-symmetric path.
 */
import { memo, type ReactElement } from "react";

/**
 * Canonical asymmetric egg path used as the rest pose by every layer.
 * Wider at the bottom, slightly tapered at the top, leans right.
 */
export const EGG_PATH =
  "M 250 75 C 340 65 415 140 420 250 C 425 360 355 425 250 425 C 145 425 75 360 80 250 C 85 140 160 60 250 75 Z";

/** Four asymmetric-egg variants the path-morph cycles through. */
const PATH_VARIANT_A = EGG_PATH;
const PATH_VARIANT_B =
  "M 250 55 C 305 55 365 145 405 295 C 425 380 340 440 250 440 C 150 440 80 360 95 260 C 115 160 200 60 250 55 Z";
const PATH_VARIANT_C =
  "M 205 50 C 280 30 405 165 425 285 C 435 380 320 440 220 430 C 120 420 60 340 80 215 C 100 115 130 70 205 50 Z";
const PATH_VARIANT_D =
  "M 285 90 C 380 80 445 185 415 285 C 380 380 280 425 175 405 C 75 380 60 280 115 175 C 145 100 205 100 285 90 Z";

/**
 * Coral-dominant gradient stops. Peach occupies the top, coral the
 * largest middle band, magenta and deep-magenta the lower band.
 *
 * IMPORTANT: NO soft-pink intermediate. The v4 reference's `#ff3399`
 * speaking stop and the brief's earlier `#FF8FA8` mid stop are both
 * dropped — coral is the connective tissue (per user direction).
 */
type GradientStops = readonly { offset: string; color: string; opacity?: number }[];

const STOPS_BASE: GradientStops = [
  { offset: "0%", color: "#ffceaa" },
  { offset: "42%", color: "#ff876c" },
  { offset: "82%", color: "#e82182" },
  { offset: "100%", color: "#c9146c" },
];

const STOPS_TYPING: GradientStops = [
  { offset: "0%", color: "#ffd8b8" },
  { offset: "42%", color: "#ff876c" },
  { offset: "82%", color: "#e82182" },
  { offset: "100%", color: "#c9146c" },
];

const STOPS_SPEAKING: GradientStops = [
  { offset: "0%", color: "#ffe1cc" },
  { offset: "50%", color: "#ff876c" },
  { offset: "85%", color: "#e82182" },
  { offset: "100%", color: "#c9146c" },
];

function gradientStopElements(stops: GradientStops): ReactElement[] {
  return stops.map((s) => (
    <stop
      key={s.offset}
      offset={s.offset}
      stopColor={s.color}
      stopOpacity={s.opacity ?? 1}
    />
  ));
}

/**
 * Standard radial-gradient block with animated focal point. Reused by
 * every layer with per-state stop set and timing.
 */
function AnimatedGradient({
  id,
  stops,
  duration,
  fxValues = "35%; 45%; 30%; 40%; 35%",
  fyValues = "30%; 45%; 45%; 30%; 30%",
}: {
  id: string;
  stops: GradientStops;
  duration: string;
  fxValues?: string;
  fyValues?: string;
}): ReactElement {
  return (
    <radialGradient id={id} cx="50%" cy="50%" r="75%">
      <animate
        attributeName="fx"
        calcMode="spline"
        keySplines="0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1"
        values={fxValues}
        dur={duration}
        repeatCount="indefinite"
      />
      <animate
        attributeName="fy"
        calcMode="spline"
        keySplines="0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1"
        values={fyValues}
        dur={duration}
        repeatCount="indefinite"
      />
      {gradientStopElements(stops)}
    </radialGradient>
  );
}

/**
 * Edge-mask gradient that drifts the bright highlight across the orb to
 * simulate breathing light. Same shape for every layer with different
 * timing.
 */
function AnimatedEdgeMask({
  id,
  duration,
  cxValues = "400; 250; 100; 250; 400",
  cyValues = "250; 400; 250; 100; 250",
}: {
  id: string;
  duration: string;
  cxValues?: string;
  cyValues?: string;
}): ReactElement {
  return (
    <radialGradient id={id} cx="250" cy="250" r="280" gradientUnits="userSpaceOnUse">
      <animate
        attributeName="cx"
        calcMode="spline"
        keySplines="0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1"
        values={cxValues}
        dur={duration}
        repeatCount="indefinite"
      />
      <animate
        attributeName="cy"
        calcMode="spline"
        keySplines="0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1; 0.4 0 0.6 1"
        values={cyValues}
        dur={duration}
        repeatCount="indefinite"
      />
      <stop offset="0%" stopColor="white" />
      <stop offset="35%" stopColor="white" />
      <stop offset="100%" stopColor="black" />
    </radialGradient>
  );
}

/**
 * Render the path-morph animation. Each layer has its own duration and
 * variant ordering; the shape always lands back on `EGG_PATH` at the
 * cycle boundary so layer crossfades blend cleanly.
 */
function MorphPath({ id, duration }: { id: string; duration: string }): ReactElement {
  return (
    <path id={id}>
      <animate
        attributeName="d"
        dur={duration}
        repeatCount="indefinite"
        calcMode="spline"
        keySplines="0.45 0 0.25 1; 0.45 0 0.25 1; 0.45 0 0.25 1; 0.45 0 0.25 1"
        keyTimes="0; 0.25; 0.5; 0.75; 1"
        values={`${PATH_VARIANT_A}; ${PATH_VARIANT_B}; ${PATH_VARIANT_C}; ${PATH_VARIANT_D}; ${PATH_VARIANT_A}`}
      />
    </path>
  );
}

/* ── IDLE layer — slow breath, 8s loop ─────────────────────────────── */

export const IdleLayer = memo(function IdleLayer(): ReactElement {
  return (
    <svg
      className="spren-svg"
      viewBox="-50 -50 600 600"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <MorphPath id="idle-path" duration="8s" />
        <AnimatedGradient id="idle-grad" stops={STOPS_BASE} duration="8s" />
        <filter id="idle-deep-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="55" />
        </filter>
        <filter id="idle-mid-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="25" />
        </filter>
        <filter id="idle-core-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" />
        </filter>
        <AnimatedEdgeMask id="idle-mask" duration="8s" />
        <mask id="idle-edge-mask">
          <rect x="-100" y="-100" width="700" height="700" fill="url(#idle-mask)" />
        </mask>
      </defs>
      <use href="#idle-path" fill="url(#idle-grad)" filter="url(#idle-deep-blur)" opacity="0.65" />
      <use href="#idle-path" fill="url(#idle-grad)" filter="url(#idle-mid-blur)" opacity="0.85" />
      <use href="#idle-path" fill="url(#idle-grad)" filter="url(#idle-core-blur)" mask="url(#idle-edge-mask)" />
    </svg>
  );
});

/* ── TYPING layer — three-dot vortex orbit, 12s loop ───────────────── */

export const TypingLayer = memo(function TypingLayer(): ReactElement {
  return (
    <div className="spren-vortex-wrapper">
      <svg
        className="spren-svg"
        viewBox="-50 -50 600 600"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <defs>
          <MorphPath id="type-path-1" duration="6s" />
          <MorphPath id="type-path-2" duration="3s" />
          <MorphPath id="type-path-3" duration="2s" />
          <AnimatedGradient
            id="type-grad"
            stops={STOPS_TYPING}
            duration="12s"
            fxValues="25%; 75%; 25%; 75%; 25%"
            fyValues="40%; 60%; 40%; 60%; 40%"
          />
          <filter id="type-deep-blur" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="55">
              <animate
                attributeName="stdDeviation"
                dur="12s"
                repeatCount="indefinite"
                calcMode="spline"
                keySplines="0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1"
                keyTimes="0; 0.75; 0.80; 0.85; 1"
                values="55; 55; 140; 55; 55"
              />
            </feGaussianBlur>
          </filter>
          <filter id="type-mid-blur" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="25" />
          </filter>
          <filter id="type-gooey-split" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="16" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 25 -10"
              result="goo"
            />
            <feComposite in="SourceGraphic" in2="goo" operator="atop" result="composite" />
            <feGaussianBlur in="composite" stdDeviation="3" />
          </filter>
          <radialGradient id="type-mask" cx="250" cy="250" r="500" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="white" />
            <stop offset="60%" stopColor="white" />
            <stop offset="100%" stopColor="black" />
          </radialGradient>
          <mask id="type-edge-mask">
            <rect x="-500" y="-500" width="1500" height="1500" fill="url(#type-mask)" />
          </mask>
        </defs>

        <g opacity="0.65">
          <g className="spren-type-dot spren-type-dot-1">
            <use href="#type-path-1" fill="url(#type-grad)" filter="url(#type-deep-blur)" />
          </g>
          <g className="spren-type-dot spren-type-dot-2">
            <use href="#type-path-2" fill="url(#type-grad)" filter="url(#type-deep-blur)" />
          </g>
          <g className="spren-type-dot spren-type-dot-3">
            <use href="#type-path-3" fill="url(#type-grad)" filter="url(#type-deep-blur)" />
          </g>
        </g>

        <g opacity="0.85">
          <g className="spren-type-dot spren-type-dot-1">
            <use href="#type-path-1" fill="url(#type-grad)" filter="url(#type-mid-blur)" />
          </g>
          <g className="spren-type-dot spren-type-dot-2">
            <use href="#type-path-2" fill="url(#type-grad)" filter="url(#type-mid-blur)" />
          </g>
          <g className="spren-type-dot spren-type-dot-3">
            <use href="#type-path-3" fill="url(#type-grad)" filter="url(#type-mid-blur)" />
          </g>
        </g>

        <g filter="url(#type-gooey-split)" mask="url(#type-edge-mask)">
          <g className="spren-type-dot spren-type-dot-1">
            <use href="#type-path-1" fill="url(#type-grad)" />
          </g>
          <g className="spren-type-dot spren-type-dot-2">
            <use href="#type-path-2" fill="url(#type-grad)" />
          </g>
          <g className="spren-type-dot spren-type-dot-3">
            <use href="#type-path-3" fill="url(#type-grad)" />
          </g>
        </g>
      </svg>
    </div>
  );
});

/* ── THINKING layer — rapid pulse + slight horizontal shake, 2s loop ── */

export const ThinkingLayer = memo(function ThinkingLayer(): ReactElement {
  return (
    <svg
      className="spren-svg"
      viewBox="-50 -50 600 600"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <path id="think-path">
          <animate
            attributeName="d"
            dur="3s"
            repeatCount="indefinite"
            calcMode="spline"
            keySplines="0.45 0 0.25 1; 0.45 0 0.25 1"
            keyTimes="0; 0.5; 1"
            values={`${EGG_PATH}; ${PATH_VARIANT_B}; ${EGG_PATH}`}
          />
        </path>
        <AnimatedGradient
          id="think-grad"
          stops={STOPS_BASE}
          duration="1.5s"
          fxValues="30%; 50%; 70%; 50%; 30%"
          fyValues="50%; 30%; 50%; 70%; 50%"
        />
        <filter id="think-deep-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="35" />
        </filter>
        <filter id="think-mid-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="15" />
        </filter>
        <filter id="think-core-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" />
        </filter>
        <AnimatedEdgeMask
          id="think-mask"
          duration="1.5s"
          cxValues="100; 250; 400; 250; 100"
          cyValues="250; 100; 250; 400; 250"
        />
        <mask id="think-edge-mask">
          <rect x="-100" y="-100" width="700" height="700" fill="url(#think-mask)" />
        </mask>
      </defs>
      <g>
        <animateTransform
          attributeName="transform"
          type="translate"
          values="-5,0; 5,0; -5,0"
          dur="1s"
          repeatCount="indefinite"
        />
        <use href="#think-path" fill="url(#think-grad)" filter="url(#think-deep-blur)" opacity="0.65" />
      </g>
      <g>
        <animateTransform
          attributeName="transform"
          type="translate"
          values="0,-3; 0,3; 0,-3"
          dur="0.8s"
          repeatCount="indefinite"
        />
        <use href="#think-path" fill="url(#think-grad)" filter="url(#think-mid-blur)" opacity="0.85" />
      </g>
      <use href="#think-path" fill="url(#think-grad)" filter="url(#think-core-blur)" mask="url(#think-edge-mask)" />
    </svg>
  );
});

/* ── SPEAKING layer — amplitude pump + slow drift, 8s + 1.8s ────────── */

export const SpeakingLayer = memo(function SpeakingLayer(): ReactElement {
  return (
    <svg
      className="spren-svg"
      viewBox="-50 -50 600 600"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <MorphPath id="speak-path" duration="8s" />
        <AnimatedGradient id="speak-grad" stops={STOPS_SPEAKING} duration="8s" />
        <filter id="speak-deep-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="60" />
        </filter>
        <filter id="speak-mid-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence type="fractalNoise" baseFrequency="0.01" numOctaves="1" result="noise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale="0"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          >
            <animate
              attributeName="scale"
              dur="1.8s"
              repeatCount="indefinite"
              calcMode="spline"
              keySplines="0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1"
              keyTimes="0; 0.12; 0.25; 0.4; 0.55; 0.65; 0.75; 0.85; 0.95; 1"
              values="0; 90; 45; 120; 0; 0; 80; 35; 0; 0"
            />
          </feDisplacementMap>
          <feGaussianBlur in="displaced" stdDeviation="30" />
        </filter>
        <filter id="speak-core-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence type="fractalNoise" baseFrequency="0.01" numOctaves="1" result="noise">
            <animate
              attributeName="baseFrequency"
              values="0.01; 0.015; 0.01"
              dur="1.8s"
              repeatCount="indefinite"
            />
          </feTurbulence>
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale="0"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          >
            <animate
              attributeName="scale"
              dur="1.8s"
              repeatCount="indefinite"
              calcMode="spline"
              keySplines="0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1; 0.4 0 0.2 1"
              keyTimes="0; 0.12; 0.25; 0.4; 0.55; 0.65; 0.75; 0.85; 0.95; 1"
              values="0; 65; 25; 85; 0; 0; 55; 20; 0; 0"
            />
          </feDisplacementMap>
          <feGaussianBlur in="displaced" stdDeviation="3" />
        </filter>
        <AnimatedEdgeMask id="speak-mask" duration="8s" />
        <mask id="speak-edge-mask">
          <rect x="-100" y="-100" width="700" height="700" fill="url(#speak-mask)" />
        </mask>
      </defs>
      <use href="#speak-path" fill="url(#speak-grad)" filter="url(#speak-deep-blur)" opacity="0.75" />
      <use href="#speak-path" fill="url(#speak-grad)" filter="url(#speak-mid-blur)" opacity="0.85" />
      <use href="#speak-path" fill="url(#speak-grad)" filter="url(#speak-core-blur)" mask="url(#speak-edge-mask)" />
    </svg>
  );
});
