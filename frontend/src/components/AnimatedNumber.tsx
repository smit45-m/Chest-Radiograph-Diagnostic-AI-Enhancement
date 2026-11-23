import React, { useEffect, useRef, useState } from "react";

interface AnimatedNumberProps {
  value: number | null | undefined;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  fallback?: string;
  durationMs?: number;
  className?: string;
  style?: React.CSSProperties;
}

/** Smooth Expo-Out tabular counter for clinical metrics (CTR, probabilities, latencies). */
export const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  decimals = 1,
  suffix = "",
  prefix = "",
  fallback = "—",
  durationMs = 620,
  className,
  style,
}) => {
  const [display, setDisplay] = useState<number | null>(value ?? null);
  const prevRef = useRef<number>(value ?? 0);

  useEffect(() => {
    if (value == null || Number.isNaN(value)) {
      setDisplay(null);
      return;
    }

    const startVal = prevRef.current;
    const delta = value - startVal;
    if (Math.abs(delta) < 1e-6) {
      setDisplay(value);
      return;
    }

    const startTime = performance.now();
    let rafId = 0;

    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / durationMs);
      // Smooth Expo-Out easing
      const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
      const current = startVal + delta * eased;
      setDisplay(current);
      if (t < 1) {
        rafId = requestAnimationFrame(step);
      } else {
        prevRef.current = value;
      }
    };

    rafId = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(rafId);
      prevRef.current = value;
    };
  }, [value, durationMs]);

  if (display == null) {
    return (
      <span className={className} style={style}>
        {fallback}
      </span>
    );
  }

  return (
    <span className={className} style={style}>
      {prefix}
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
};
