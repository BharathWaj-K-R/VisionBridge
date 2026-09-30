import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";

export const QUICK_ACCESS_SLOTS = 10;
type QuickAccessContextValue = {
  slots: Array<string | null>;
  loading: boolean;
  assign: (slot: number, phrase: string | null) => Promise<void>;
  move: (from: number, to: number) => Promise<void>;
  clear: (slot: number) => Promise<void>;
};

const QuickAccessContext = createContext<QuickAccessContextValue | null>(null);

function normalize(slots: Array<string | null>): Array<string | null> {
  return Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => slots[index] || null);
}

export function QuickAccessProvider({ children }: { children: ReactNode }) {
  const [slots, setSlots] = useState<Array<string | null>>(() => Array(QUICK_ACCESS_SLOTS).fill(null));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    api.quickAccess()
      .then((payload) => { if (mounted) setSlots(normalize(payload.slots)); })
      .catch(() => undefined)
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, []);

  const assign = async (slot: number, phrase: string | null) => {
    if (slot < 0 || slot >= QUICK_ACCESS_SLOTS) return;
    const next = [...slots];
    next[slot] = phrase?.trim() || null;
    await api.saveQuickAccess(next);
    setSlots(next);
  };

  const move = async (from: number, to: number) => {
    if (from < 0 || from >= QUICK_ACCESS_SLOTS || to < 0 || to >= QUICK_ACCESS_SLOTS) return;
    const next = [...slots];
    [next[from], next[to]] = [next[to], next[from]];
    await api.saveQuickAccess(next);
    setSlots(next);
  };

  const clear = async (slot: number) => assign(slot, null);

  const value = useMemo(() => ({ slots, loading, assign, move, clear }), [slots, loading]);
  return <QuickAccessContext.Provider value={value}>{children}</QuickAccessContext.Provider>;
}

export function useQuickAccess() {
  const value = useContext(QuickAccessContext);
  if (!value) throw new Error("useQuickAccess must be used inside QuickAccessProvider");
  return value;
}
