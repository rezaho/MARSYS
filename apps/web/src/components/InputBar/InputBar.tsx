/**
 * The Spren conversation input bar.
 *
 * White surface, 24px border-radius, magenta focus ring. The circular
 * send button is `--ink` by default, transitions to `--magenta` on
 * hover, and is disabled (opacity 0.4, `not-allowed` cursor) when the
 * input is empty.
 *
 * Renders a single-row `<textarea>` so Shift+Enter can insert a newline
 * (Enter on its own submits). The textarea expands a little when the
 * content wraps; CSS clamps the visible height.
 */
import { useEffect, useRef, useState, type ReactElement } from "react";

import "./InputBar.css";

interface InputBarProps {
  placeholder?: string;
  onSubmit: (text: string) => void;
  /**
   * Fired with `true` when the input gains focus, `false` when it
   * loses focus. The home page and chat sheet use this to drive the
   * orb's `state` between idle and typing.
   */
  onFocusChange?: (focused: boolean) => void;
  /** Optional id for assistive tech / E2E. */
  inputId?: string;
  /** Forwarded test id (default `input-bar`). */
  testId?: string;
  /** Autofocus on mount. */
  autoFocus?: boolean;
}

export function InputBar({
  placeholder = "What's on your mind?",
  onSubmit,
  onFocusChange,
  inputId,
  testId = "input-bar",
  autoFocus = false,
}: InputBarProps): ReactElement {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (autoFocus) {
      inputRef.current?.focus();
    }
  }, [autoFocus]);

  const empty = value.trim().length === 0;

  function submit() {
    if (empty) return;
    onSubmit(value);
    setValue("");
  }

  return (
    <form
      className="input-bar"
      data-testid={testId}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={inputRef}
        id={inputId}
        rows={1}
        autoComplete="off"
        placeholder={placeholder}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onFocus={() => onFocusChange?.(true)}
        onBlur={() => onFocusChange?.(false)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            inputRef.current?.blur();
            return;
          }
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        data-testid={`${testId}-input`}
      />
      <button
        type="submit"
        className="input-bar-send"
        aria-label="Send message"
        disabled={empty}
        data-testid={`${testId}-send`}
      >
        <svg viewBox="0 0 24 24" width={16} height={16} aria-hidden="true">
          <path
            d="M5 12 L19 12 M13 6 L19 12 L13 18"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </form>
  );
}
