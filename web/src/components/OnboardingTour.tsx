"use client";

import { useEffect, useState } from 'react';
import Joyride, { CallBackProps, STATUS, Step } from 'react-joyride';

const TOUR_KEY = 'tour_seen_v1';

export const OnboardingTour = () => {
  const [run, setRun] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    const seen = localStorage.getItem(TOUR_KEY);
    if (!seen) {
      setRun(true);
    }

    // Listen for manual trigger
    const handleManualTrigger = () => {
      setRun(true);
    };
    window.addEventListener('trigger-onboarding-tour', handleManualTrigger);

    return () => {
      window.removeEventListener('trigger-onboarding-tour', handleManualTrigger);
    };
  }, []);

  const handleCallback = (data: CallBackProps) => {
    const { status } = data;
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
      setRun(false);
      localStorage.setItem(TOUR_KEY, 'true');
    }
  };

  const steps: Step[] = [
    {
      target: '[data-tour-id="activity-ask-mentor"]',
      content: 'Click the sparkles to open your AI research mentor. Ask questions, search papers, and get help.',
      placement: 'right',
      disableBeacon: true,
    },
    {
      target: '[data-tour-id="view-notebook"]',
      content: 'Switch to the Notebook view to write your paper. It supports Markdown, rich text, and exports.',
      placement: 'right',
    },
    {
      target: '[data-tour-id="notebook-toolbar"]',
      content: 'Format your text, add images, and export your work using the floating toolbar.',
      placement: 'bottom',
    },
    {
      target: '[data-tour-id="sidebar-tabs"]',
      content: 'Toggle between your document Context (uploaded files) and your saved Notes.',
      placement: 'right',
    },
    {
      target: '[data-tour-id="upload-dropzone"]',
      content: 'Upload PDFs or drag & drop files here to add them to your research context.',
      placement: 'right',
    },
    {
      target: '[data-tour-id="chat-toolcalls"]',
      content: 'Your mentor needs to be open for this one! Chat responses and tool outputs (like search results) appear here.',
      placement: 'left',
    },
  ];

  if (!isMounted) return null;

  return (
    <Joyride
      steps={steps}
      run={run}
      continuous
      showProgress
      showSkipButton
      callback={handleCallback}
      styles={{
        options: {
          primaryColor: '#E69F00', // Okabe-Ito Orange
          backgroundColor: '#ffffff',
          textColor: '#1c1917',
          arrowColor: '#ffffff',
          zIndex: 1000,
        },
        tooltip: {
          borderRadius: '0.75rem',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
          fontSize: '14px',
          padding: '16px',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        },
        buttonNext: {
          borderRadius: '0.5rem',
          fontSize: '12px',
          fontWeight: 600,
          padding: '8px 16px',
          fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        },
        buttonBack: {
          color: '#78716c',
          marginRight: '10px',
          fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        },
        buttonSkip: {
          color: '#a8a29e',
          fontSize: '12px',
          fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        },
      }}
    />
  );
};
