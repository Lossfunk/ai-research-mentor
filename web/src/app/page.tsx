"use client";

import { useState, useEffect } from "react";
import LandingPage from "@/components/Home/LandingPage";
import WorkspaceLayout from "@/components/Workspace/WorkspaceLayout";

export default function Home() {
  const [viewState, setViewState] = useState<'landing' | 'workspace'>('landing');
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    const hasSeenLanding = localStorage.getItem('landing_seen_v1');
    if (hasSeenLanding) {
      setViewState('workspace');
    }
  }, []);

  const handleEnter = () => {
    localStorage.setItem('landing_seen_v1', 'true');
    setViewState('workspace');
  };

  if (!isMounted) return null; // Prevent hydration mismatch

  return (
    <>
      {viewState === 'landing' && (
        <LandingPage onEnter={handleEnter} />
      )}
      
      {viewState === 'workspace' && (
        <WorkspaceLayout />
      )}
    </>
  );
}
