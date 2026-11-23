import React, { useRef } from "react";

interface SpotlightCardProps extends React.HTMLAttributes<HTMLDivElement> {
  strong?: boolean;
  children: React.ReactNode;
}

/**
 * Architectural clinical surface with tactile cursor-tracked radial light sheen.
 * Uses CSS custom properties directly on DOM ref for zero React re-renders during mousemove.
 */
export const SpotlightCard: React.FC<SpotlightCardProps> = ({
  strong = false,
  className = "",
  style,
  children,
  ...rest
}) => {
  const ref = useRef<HTMLDivElement | null>(null);

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - rect.left}px`);
    el.style.setProperty("--my", `${e.clientY - rect.top}px`);
    el.style.setProperty("--spot-opacity", "1");
    rest.onMouseMove?.(e);
  };

  const handleLeave = (e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (el) el.style.setProperty("--spot-opacity", "0");
    rest.onMouseLeave?.(e);
  };

  return (
    <div
      ref={ref}
      {...rest}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      className={`${strong ? "atelier-card-strong" : "atelier-card"} spotlight-surface ${className}`}
      style={style}
    >
      {children}
    </div>
  );
};
