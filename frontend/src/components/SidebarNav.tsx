import React from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Database,
  FileText,
  Layers,
  LayoutGrid,
  Scan,
  Sparkles,
  UserRound,
} from "lucide-react";
import { NAV_PAGES, type PageId } from "../data/clinicalCatalog";
import type { AIConfigStatus, HealthStatus, UserRecord } from "../types/clinical";

interface SidebarNavProps {
  activePage: PageId;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onSelectPage: (page: PageId) => void;
  health: HealthStatus | null;
  aiConfig: AIConfigStatus | null;
  users: UserRecord[];
  activeUserId: string | null;
  onSelectUser: (userId: string) => void;
  positiveCount: number;
  decisionActive: boolean;
  hasReport: boolean;
  batchStatCount: number | null;
  totalUserStudies: number;
  onOpenGemini: () => void;
}

const PAGE_ICONS: Record<
  PageId,
  React.ComponentType<{ className?: string; style?: React.CSSProperties }>
> = {
  overview: LayoutGrid,
  studio: Scan,
  consensus: Activity,
  report: FileText,
  triage: Layers,
  records: Database,
};

export const SidebarNav: React.FC<SidebarNavProps> = ({
  activePage,
  collapsed,
  onToggleCollapse,
  onSelectPage,
  health,
  aiConfig,
  users,
  activeUserId,
  onSelectUser,
  positiveCount,
  batchStatCount,
  totalUserStudies,
  onOpenGemini,
}) => {
  const isCuda = health?.device?.includes("cuda");
  const geminiReady = Boolean(aiConfig?.gemini_configured || health?.gemini_configured);
  const activeUser = users.find((u) => u.id === activeUserId) || users[0] || null;

  return (
    <>
      {/* Desktop Whisper-Clean Clinical Rail */}
      <motion.aside
        aria-label="Primary clinical workspace navigation"
        animate={{ width: collapsed ? 72 : 228 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="no-print fixed top-0 bottom-0 left-0 z-40 hidden flex-col border-r lg:flex"
        style={{
          backgroundColor: "var(--bg-glass)",
          borderColor: "var(--border-subtle)",
          backdropFilter: "blur(24px) saturate(160%)",
          WebkitBackdropFilter: "blur(24px) saturate(160%)",
        }}
      >
        {/* Brand Row */}
        <div
          className="flex h-14 shrink-0 items-center justify-between border-b px-3.5"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          <button
            type="button"
            onClick={() => onSelectPage("overview")}
            className="group flex min-w-0 cursor-pointer items-center gap-2.5 text-left"
          >
            <div
              className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border"
              style={{
                borderColor: "var(--border-strong)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <svg
                viewBox="0 0 28 28"
                className="h-4.5 w-4.5 transition-transform duration-300 group-hover:scale-105"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M4.5 14.5h4.2l2.3-5.2 3.2 10 2.3-4.8h7"
                  stroke="var(--accent-clinical)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            {!collapsed && (
              <div className="min-w-0">
                <div className="truncate font-display text-[13.5px] font-bold tracking-[0.02em]">
                  CXR·Diagnostic
                </div>
              </div>
            )}
          </button>

          {!collapsed && (
            <button
              type="button"
              onClick={onToggleCollapse}
              title="Collapse sidebar"
              aria-label="Collapse navigation rail"
              className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg transition-colors hover:bg-teal-500/10"
              style={{ color: "var(--text-muted)" }}
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Active Patient Switcher (Clean & Single-Row) */}
        {!collapsed && activeUser && (
          <div
            className="border-b px-3 py-3"
            style={{ borderColor: "var(--border-subtle)" }}
          >
            <div
              className="mb-1.5 px-1 font-mono-tech text-[9.5px] tracking-[0.14em] uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              Patient
            </div>
            <div
              className="flex items-center gap-2 rounded-xl border px-2.5 py-2 transition-colors focus-within:border-teal-500/50"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <UserRound
                className="h-3.5 w-3.5 shrink-0"
                style={{ color: "var(--accent-clinical)" }}
              />
              <select
                aria-label="Active Patient or User Record"
                value={activeUser.id}
                onChange={(e) => onSelectUser(e.target.value)}
                className="w-full cursor-pointer truncate bg-transparent text-xs font-medium focus:outline-none"
                style={{ color: "var(--text-primary)" }}
              >
                {users.map((u) => (
                  <option
                    key={u.id}
                    value={u.id}
                    style={{
                      backgroundColor: "var(--bg-elevated)",
                      color: "var(--text-primary)",
                    }}
                  >
                    {u.full_name} ({u.mrn})
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* Clean Single-Line Navigation Links */}
        <nav aria-label="Workspace pages" className="flex-1 space-y-1 overflow-y-auto px-2.5 py-3">
          {NAV_PAGES.map((page) => {
            const Icon = PAGE_ICONS[page.id];
            const isActive = activePage === page.id;

            let badge: { text: string; alert?: boolean } | null = null;
            if (page.id === "studio" && positiveCount > 0) {
              badge = { text: `${positiveCount}`, alert: true };
            } else if (page.id === "triage" && batchStatCount != null && batchStatCount > 0) {
              badge = { text: `${batchStatCount}`, alert: true };
            } else if (page.id === "records" && totalUserStudies > 0) {
              badge = { text: `${totalUserStudies}` };
            }

            return (
              <button
                key={page.id}
                type="button"
                onClick={() => onSelectPage(page.id)}
                title={collapsed ? `${page.label} (${page.keyHint})` : page.shortDesc}
                className={`group relative flex w-full cursor-pointer items-center rounded-xl transition-colors ${
                  collapsed ? "justify-center px-0 py-2.5" : "gap-2.5 px-3 py-2.5 text-left"
                }`}
                style={{
                  color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                {isActive && (
                  <motion.span
                    layoutId="sidebar-active-surface"
                    transition={{ type: "spring", stiffness: 420, damping: 34 }}
                    className="absolute inset-0 -z-10 rounded-xl"
                    style={{
                      backgroundColor: "var(--accent-clinical-dim)",
                    }}
                  />
                )}

                <Icon
                  className="h-4 w-4 shrink-0 transition-transform duration-200 group-hover:scale-105"
                  style={{
                    color: isActive ? "var(--accent-clinical)" : "var(--text-muted)",
                  }}
                />

                {!collapsed && (
                  <>
                    <span className="min-w-0 flex-1 truncate text-[13px] font-medium">
                      {page.label}
                    </span>

                    {badge ? (
                      <span
                        className="rounded-full px-1.5 py-0.5 font-mono-tech text-[10px] font-semibold"
                        style={{
                          backgroundColor: badge.alert
                            ? "var(--accent-alert-dim)"
                            : "var(--accent-clinical-dim)",
                          color: badge.alert
                            ? "var(--accent-alert)"
                            : "var(--accent-clinical)",
                        }}
                      >
                        {badge.text}
                      </span>
                    ) : (
                      <span
                        className="font-mono-tech text-[10px] opacity-0 transition-opacity group-hover:opacity-60"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {page.keyHint}
                      </span>
                    )}
                  </>
                )}
              </button>
            );
          })}
        </nav>

        {/* Quiet Single-Card AI & Engine Footer */}
        <div
          className="shrink-0 border-t p-2.5"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          {collapsed ? (
            <button
              type="button"
              onClick={onToggleCollapse}
              title="Expand sidebar"
              aria-label="Expand navigation rail"
              className="mx-auto flex h-9 w-9 cursor-pointer items-center justify-center rounded-xl transition-colors hover:bg-teal-500/10"
              style={{ color: "var(--text-secondary)" }}
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onOpenGemini}
              className="flex w-full cursor-pointer items-center justify-between rounded-xl border px-3 py-2.5 text-left transition-colors hover:border-teal-500/40"
              style={{
                borderColor: "var(--border-subtle)",
                backgroundColor: "var(--bg-elevated)",
              }}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 text-xs font-medium">
                  <Sparkles
                    className="h-3.5 w-3.5 shrink-0"
                    style={{ color: geminiReady ? "var(--accent-clinical)" : "var(--text-muted)" }}
                  />
                  <span className="truncate">{geminiReady ? "Gemini AI" : "Connect AI"}</span>
                </div>
                <div
                  className="mt-0.5 truncate font-mono-tech text-[10px]"
                  style={{ color: "var(--text-muted)" }}
                >
                  {isCuda ? "RTX GPU" : "Cloud CPU"} · {geminiReady ? "Live" : "Offline"}
                </div>
              </div>
              <span
                className="h-2 w-2 shrink-0 rounded-full clinical-led"
                style={{
                  backgroundColor: geminiReady ? "var(--accent-clinical)" : "var(--text-muted)",
                }}
              />
            </button>
          )}
        </div>
      </motion.aside>

      {/* Mobile Bottom Dock */}
      <nav
        aria-label="Mobile workspace navigation"
        className="no-print fixed right-3 bottom-3 left-3 z-40 flex items-center justify-around rounded-2xl border px-1.5 py-2 lg:hidden"
        style={{
          backgroundColor: "var(--bg-glass)",
          borderColor: "var(--border-strong)",
          backdropFilter: "blur(20px)",
          boxShadow: "var(--shadow-float)",
        }}
      >
        {NAV_PAGES.map((page) => {
          const Icon = PAGE_ICONS[page.id];
          const isActive = activePage === page.id;
          return (
            <button
              key={page.id}
              type="button"
              onClick={() => onSelectPage(page.id)}
              className="flex cursor-pointer flex-col items-center gap-1 rounded-xl px-2 py-1.5 text-[10px] font-medium"
              style={{
                backgroundColor: isActive ? "var(--accent-clinical-dim)" : "transparent",
                color: isActive ? "var(--accent-clinical)" : "var(--text-secondary)",
              }}
            >
              <Icon className="h-4 w-4" />
              <span className="truncate">{page.label.split(" ")[0]}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
};
