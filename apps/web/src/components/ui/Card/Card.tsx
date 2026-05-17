/**
 * Card primitive — surface tile with the design-system's rounded
 * radius + rule border.
 *
 * `interactive` adds the hover behavior used by the workflow list
 * cards: border darkens to ink-faint at 55% alpha, a 1px lift on the
 * Y axis, animated through the spring easing curve.
 *
 * `as` lets the consumer render as something other than div — TanStack
 * Router's `<Link>` for navigable cards, `<button>` for actionable
 * surfaces. Passes all unknown props through.
 *
 * `padding`: `sm` (12px), `md` (16px 18px — the workflow card shape).
 * No `lg` until a use case arrives.
 */
import {
  forwardRef,
  type ElementType,
  type HTMLAttributes,
  type ReactNode,
  type Ref,
} from "react";

import "./Card.css";

export type CardPadding = "sm" | "md";

interface CardOwnProps {
  as?: ElementType;
  interactive?: boolean;
  padding?: CardPadding;
  className?: string;
  children?: ReactNode;
}

// Allow consumers to pass any HTML attribute the underlying element
// accepts (href when `as="a"`, type when `as="button"`, etc.). Typed as
// HTMLAttributes for IDE support; we don't narrow per-`as` here — the
// design system's three consumers don't need that ceremony.
export type CardProps = CardOwnProps &
  Omit<HTMLAttributes<HTMLElement>, keyof CardOwnProps> &
  Record<string, unknown>;

export const Card = forwardRef<HTMLElement, CardProps>(function Card(
  {
    as = "div",
    interactive = false,
    padding = "md",
    className,
    children,
    ...rest
  },
  ref,
) {
  const classes = [
    "ui-card",
    `ui-card--padding-${padding}`,
    interactive ? "ui-card--interactive" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  const Tag = as as ElementType;
  return (
    <Tag ref={ref as Ref<HTMLElement>} className={classes} {...rest}>
      {children}
    </Tag>
  );
});
