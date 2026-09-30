import { createContext, useContext, useMemo, type ReactNode } from "react";
import { usePersonalization } from "./PersonalizationContext";

export const QUICK_ACCESS_SLOTS = 10;

type QuickAccessContextValue = {
  slots: Array<string | null>;
  loading: boolean;
  assign: (slot: number, phrase: string | null) => Promise<void>;
  move: (from: number, to: number) => Promise<void>;
  clear: (slot: number) => Promise<void>;
};

const QuickAccessContext = createContext<QuickAccessContextValue | null>(null);

export function QuickAccessProvider({ children }: { children: ReactNode }) {
  const { activeProfile, loading, updateConfig } = usePersonalization();
  const slots = Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => activeProfile?.config.quickAccess[index] || null);

  const assign = async (slot: number, phrase: string | null) => {
    if (slot < 0 || slot >= QUICK_ACCESS_SLOTS) return;
    const next = [...slots];
    next[slot] = phrase?.trim() || null;
    await updateConfig({ quickAccess: next });
  };

  const move = async (from: number, to: number) => {
    if (from < 0 || from >= QUICK_ACCESS_SLOTS || to < 0 || to >= QUICK_ACCESS_SLOTS) return;
    const next = [...slots];
    [next[from], next[to]] = [next[to], next[from]];
    await updateConfig({ quickAccess: next });
  };

  const clear = async (slot: number) => assign(slot, null);

  const value = useMemo(() => ({ slots, loading, assign, move, clear }), [slots.join("|"), loading, activeProfile?.id]);
  return <QuickAccessContext.Provider value={value}>{children}</QuickAccessContext.Provider>;
}

export function useQuickAccess() {
  const value = useContext(QuickAccessContext);
  if (!value) throw new Error("useQuickAccess must be used inside QuickAccessProvider");
  return value;
}
