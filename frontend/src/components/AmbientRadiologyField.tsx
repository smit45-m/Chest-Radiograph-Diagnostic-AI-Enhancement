import React, { useEffect, useRef } from "react";
import type { ThemeMode } from "../types/clinical";

interface AmbientRadiologyFieldProps {
  theme: ThemeMode;
}

interface GridPulse {
  horizontal: boolean;
  track: number;
  pos: number;
  speed: number;
  length: number;
  alpha: number;
  tint: "teal" | "ice";
}

/**
 * Full-viewport GPU-friendly atmospheric radiology canvas:
 * - Precision calibrated 64px DICOM coordinate grid
 * - Subtle optical '+' fiducials at every 3rd intersection that softly illuminate near the pointer
 * - Slow traveling radiographic light pulses gliding along coordinate tracks
 * - Soft breathing volumetric pulmonary aura gradients (CSS composited)
 */
export const AmbientRadiologyField: React.FC<AmbientRadiologyFieldProps> = ({ theme }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let rafId = 0;
    let width = window.innerWidth;
    let height = window.innerHeight;
    const cell = 64;

    const pointer = { x: -1000, y: -1000, tx: -1000, ty: -1000 };
    const onMouseMove = (e: MouseEvent) => {
      pointer.tx = e.clientX;
      pointer.ty = e.clientY;
    };
    const onMouseLeave = () => {
      pointer.tx = -1000;
      pointer.ty = -1000;
    };
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("mouseleave", onMouseLeave);

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    // Spawn 7 gentle traveling calibration light beams
    const spawnPulse = (initial = false): GridPulse => {
      const horizontal = Math.random() > 0.42;
      const maxTrack = Math.max(2, Math.floor((horizontal ? height : width) / cell));
      return {
        horizontal,
        track: Math.floor(Math.random() * maxTrack) * cell,
        pos: initial ? Math.random() * (horizontal ? width : height) : -140,
        speed: 0.45 + Math.random() * 0.75,
        length: 90 + Math.random() * 110,
        alpha: 0.14 + Math.random() * 0.16,
        tint: Math.random() > 0.35 ? "teal" : "ice",
      };
    };

    const pulses: GridPulse[] = Array.from({ length: 7 }, () => spawnPulse(true));

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Smooth spring lerp toward pointer
      pointer.x += (pointer.tx - pointer.x) * 0.08;
      pointer.y += (pointer.ty - pointer.y) * 0.08;

      const isDark = theme === "dark";
      const gridStroke = isDark ? "rgba(186, 218, 238, 0.032)" : "rgba(12, 22, 31, 0.038)";
      const fiducialBase = isDark ? "rgba(186, 218, 238, 0.085)" : "rgba(12, 22, 31, 0.08)";
      const tealRgb = isDark ? "20, 184, 166" : "13, 148, 136";
      const iceRgb = isDark ? "56, 189, 248" : "2, 132, 199";

      // 1. Hairline Calibrated Coordinate Grid
      ctx.beginPath();
      for (let x = 0; x <= width; x += cell) {
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, height);
      }
      for (let y = 0; y <= height; y += cell) {
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(width, y + 0.5);
      }
      ctx.strokeStyle = gridStroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      // 2. Optical '+' Fiducial Crosshairs (brighten subtly under cursor spotlight)
      const majorStep = cell * 3;
      for (let x = majorStep; x < width; x += majorStep) {
        for (let y = majorStep; y < height; y += majorStep) {
          const dist = Math.hypot(x - pointer.x, y - pointer.y);
          const prox = dist < 240 ? 1 - dist / 240 : 0;
          const arm = 3.5 + prox * 2.5;
          ctx.beginPath();
          ctx.moveTo(x - arm, y + 0.5);
          ctx.lineTo(x + arm, y + 0.5);
          ctx.moveTo(x + 0.5, y - arm);
          ctx.lineTo(x + 0.5, y + arm);
          ctx.strokeStyle =
            prox > 0.05
              ? `rgba(${tealRgb}, ${(0.12 + prox * 0.42).toFixed(3)})`
              : fiducialBase;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // 3. Traveling Optical Calibration Beams along grid tracks
      if (!prefersReduced) {
        pulses.forEach((p, idx) => {
          p.pos += p.speed;
          const limit = (p.horizontal ? width : height) + p.length + 60;
          if (p.pos > limit) {
            pulses[idx] = spawnPulse(false);
            return;
          }

          const rgb = p.tint === "teal" ? tealRgb : iceRgb;
          const a = isDark ? p.alpha : p.alpha * 0.72;

          if (p.horizontal) {
            const grad = ctx.createLinearGradient(p.pos - p.length, p.track, p.pos, p.track);
            grad.addColorStop(0, `rgba(${rgb}, 0)`);
            grad.addColorStop(0.75, `rgba(${rgb}, ${(a * 0.65).toFixed(3)})`);
            grad.addColorStop(1, `rgba(${rgb}, ${a.toFixed(3)})`);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.pos - p.length, p.track + 0.5);
            ctx.lineTo(p.pos, p.track + 0.5);
            ctx.stroke();

            // Soft luminous head dot
            ctx.fillStyle = `rgba(${rgb}, ${(a * 1.6).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(p.pos, p.track + 0.5, 1.5, 0, Math.PI * 2);
            ctx.fill();
          } else {
            const grad = ctx.createLinearGradient(p.track, p.pos - p.length, p.track, p.pos);
            grad.addColorStop(0, `rgba(${rgb}, 0)`);
            grad.addColorStop(0.75, `rgba(${rgb}, ${(a * 0.65).toFixed(3)})`);
            grad.addColorStop(1, `rgba(${rgb}, ${a.toFixed(3)})`);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.track + 0.5, p.pos - p.length);
            ctx.lineTo(p.track + 0.5, p.pos);
            ctx.stroke();

            ctx.fillStyle = `rgba(${rgb}, ${(a * 1.6).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(p.track + 0.5, p.pos, 1.5, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }

      // 4. Whisper-soft cursor illumination halo
      if (pointer.x > -200 && pointer.y > -200) {
        const halo = ctx.createRadialGradient(
          pointer.x,
          pointer.y,
          0,
          pointer.x,
          pointer.y,
          260
        );
        halo.addColorStop(0, `rgba(${tealRgb}, ${isDark ? 0.055 : 0.045})`);
        halo.addColorStop(1, `rgba(${tealRgb}, 0)`);
        ctx.fillStyle = halo;
        ctx.fillRect(pointer.x - 260, pointer.y - 260, 520, 520);
      }

      if (!prefersReduced) {
        rafId = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseleave", onMouseLeave);
    };
  }, [theme]);

  return (
    <div
      aria-hidden="true"
      className="no-print pointer-events-none fixed inset-0 z-0 overflow-hidden"
    >
      {/* Volumetric Breathing Pulmonary Optical Auras */}
      <div
        className="aura-breathe-a absolute -top-44 -left-32 h-[540px] w-[540px] rounded-full blur-[128px]"
        style={{
          background:
            theme === "dark"
              ? "radial-gradient(circle, rgba(20, 184, 166, 0.16) 0%, transparent 70%)"
              : "radial-gradient(circle, rgba(13, 148, 136, 0.13) 0%, transparent 70%)",
        }}
      />
      <div
        className="aura-breathe-b absolute top-[28%] -right-40 h-[580px] w-[580px] rounded-full blur-[140px]"
        style={{
          background:
            theme === "dark"
              ? "radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, transparent 70%)"
              : "radial-gradient(circle, rgba(2, 132, 199, 0.1) 0%, transparent 70%)",
        }}
      />
      <div
        className="aura-breathe-a absolute -bottom-48 left-[32%] h-[500px] w-[500px] rounded-full blur-[140px]"
        style={{
          background:
            theme === "dark"
              ? "radial-gradient(circle, rgba(20, 184, 166, 0.1) 0%, transparent 70%)"
              : "radial-gradient(circle, rgba(13, 148, 136, 0.08) 0%, transparent 70%)",
        }}
      />

      {/* Precision DICOM Coordinate Grid + Traveling Optical Beams */}
      <canvas ref={canvasRef} className="h-full w-full" />
    </div>
  );
};
