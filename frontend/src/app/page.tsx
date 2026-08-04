"use client";

import { useEffect } from "react";
import { LeftPanel } from "@/components/LeftPanel";
import { PlaybackGrid } from "@/components/PlaybackGrid";
import { PlayerControls } from "@/components/PlayerControls";
import { RightPanel } from "@/components/RightPanel";
import { SensorTimeline } from "@/components/SensorTimeline";
import { TopBar } from "@/components/TopBar";
import { useStudio } from "@/store/studio";

export default function Home() {
  const init = useStudio((s) => s.init);
  const error = useStudio((s) => s.error);
  const loading = useStudio((s) => s.loading);
  const episode = useStudio((s) => s.episode);

  useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <LeftPanel />
        <main className="flex w-[55%] min-w-0 flex-col p-3">
          {error && (
            <div className="mb-2 rounded border border-bad/50 bg-bad/10 px-3 py-2 text-xs text-bad">
              {error}
            </div>
          )}
          {!episode && loading && (
            <div className="mb-2 text-xs text-mute">Loading episode…</div>
          )}
          <div className="min-h-0 flex-[1.15]">
            <PlaybackGrid />
          </div>
          <PlayerControls />
          <div className="min-h-0 flex-1">
            <SensorTimeline />
          </div>
        </main>
        <RightPanel />
      </div>
    </div>
  );
}
