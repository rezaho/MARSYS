/**
 * Design-system button primitive.
 *
 * Variants:
 *   - `primary`   — ink-filled, magenta hover. The dominant CTA shape.
 *   - `secondary` — transparent + rule border, soft ink text. Cancel /
 *                   alternative actions.
 *   - `ghost`     — transparent + no border. Inline link-ish controls.
 *   - `icon`      — 36×36 square; transparent + rule border; icon child.
 *
 * Sizes: `sm` (12px text, 6px radius) and `md` (13px text, 8px radius).
 *
 * Tone (orthogonal to variant): `danger` flips the hover state to
 * `--magenta-deep` for destructive actions. Used by the right-rail
 * "Delete node" button alongside `variant="secondary"`. Defaults to
 * `neutral`.
 *
 * Loading state: replaces children with an ellipsis and applies the
 * disabled styling. The consumer's `aria-label` (if any) stays.
 *
 * Icon variant requires `aria-label` to be set by the caller — TypeScript
 * cannot enforce this without overloads; we trust the consumer.
 *
 * Uses `forwardRef` so callers can attach refs (e.g. for focus-on-mount).
 */
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import "./Button.css";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "icon";
export type ButtonSize = "sm" | "md";
export type ButtonTone = "neutral" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  tone?: ButtonTone;
  loading?: boolean;
  children?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    size = "md",
    tone = "neutral",
    loading = false,
    disabled,
    type = "button",
    className,
    children,
    ...rest
  },
  ref,
) {
  const classes = [
    "ui-button",
    `ui-button--${variant}`,
    `ui-button--${size}`,
    tone !== "neutral" ? `ui-button--${tone}` : "",
    loading ? "is-loading" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      {...rest}
      ref={ref}
      type={type}
      className={classes}
      disabled={disabled || loading}
      data-loading={loading || undefined}
    >
      {loading ? <span aria-hidden="true">…</span> : children}
    </button>
  );
});
