"use client";

import { useEffect } from "react";
import { useStudio } from "@/store/studio";

export function PlayerControls() {
  const {
    currentTime,
    playing,
    playbackRate,
    timeline,
    setCurrentTime,
    setPlaying,
    setPlaybackRate,
  } = useStudio();

  const duration = timeline?.duration_s ?? 12;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const state = useStudio.getState();
      const dur = state.timeline?.duration_s ?? 12;
      const next = state.currentTime + 0.05 * state.playbackRate;
      if (next >= dur) {
        state.setCurrentTime(dur);
        state.setPlaying(false);
      } else {
        state.setCurrentTime(next);
      }
    }, 50);
    return () => clearInterval(id);
  }, [playing, playbackRate]);

  return (
    <div className="mt-2 flex items-center gap-3 rounded border border-line bg-panel px-3 py-2">
      <button
        type="button"
        className="rounded border border-line px-2 py-1 text-xs hover:border-accent"
        onClick={() => setCurrentTime(Math.max(0, currentTime - 0.5))}
      >
        ◀
      </button>
      <button
        type="button"
        className="rounded bg-accent px-3 py-1 text-xs font-semibold text-bg"
        onClick={() => setPlaying(!playing)}
      >
        {playing ? "⏸" : "▶"}
      </button>
      <button
        type="button"
        className="rounded border border-line px-2 py-1 text-xs hover:border-accent"
        onClick={() => setCurrentTime(Math.min(duration, currentTime + 0.5))}
      >
        ▶
      </button>
      {[0.5, 1, 2].map((r) => (
        <button
          key={r}
          type="button"
          onClick={() => setPlaybackRate(r)}
          className={`rounded px-2 py-1 text-xs ${
            playbackRate === r ? "bg-accent/20 text-accent" : "text-mute hover:text-ink"
          }`}
        >
          {r}×
        </button>
      ))}
      <input
        type="range"
        min={0}
        max={duration}
        step={0.01}
        value={currentTime}
        onChange={(e) => {
          setPlaying(false);
          setCurrentTime(Number(e.target.value));
        }}
        className="mx-2 h-1 flex-1 cursor-pointer accent-accent"
      />
      <span className="min-w-[110px] font-mono text-xs text-ink">
        Timestamp: {currentTime.toFixed(3)} s
      </span>
    </div>
  );
}
