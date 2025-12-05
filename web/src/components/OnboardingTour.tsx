"use client";

import { useEffect, useState } from 'react';
import Joyride, { ACTIONS, CallBackProps, EVENTS, STATUS, Step } from 'react-joyride';

const TOUR_KEY = 'tour_seen_v1';

export const OnboardingTour = () => {
  const [run, setRun] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    // Delay the initial check to allow Workspace to fully mount and animate in
    const timer = setTimeout(() => {
      const seen = localStorage.getItem(TOUR_KEY);
      if (!seen) {
        setRun(true);
      }
    }, 500); // Reduced to 500ms for quicker start

    // Listen for manual trigger
    const handleManualTrigger = () => {
      setRun(true);
      setStepIndex(0);
    };
    window.addEventListener('trigger-onboarding-tour', handleManualTrigger);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('trigger-onboarding-tour', handleManualTrigger);
    };
  }, []);

  // Separate effect for Mentor Interaction to avoid re-setting the timer
  useEffect(() => {
    const handleMentorOpen = () => {
      if (run && stepIndex === 0) {
         setTimeout(() => setStepIndex(1), 500); 
      }
    };
    window.addEventListener('mentor-chat-opened', handleMentorOpen);
    return () => window.removeEventListener('mentor-chat-opened', handleMentorOpen);
  }, [run, stepIndex]);

  const handleCallback = (data: CallBackProps) => {
    const { status, action, type, index } = data;
    
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
      setRun(false);
      localStorage.setItem(TOUR_KEY, 'true');
    } else if (type === EVENTS.STEP_AFTER || type === EVENTS.TARGET_NOT_FOUND) {
        // Update internal step index to match joyride's
        const nextIndex = index + (action === ACTIONS.PREV ? -1 : 1);
        setStepIndex(nextIndex);
    }
  };

  const steps: Step[] = [
    {
      target: '[data-tour-id="activity-ask-mentor"]',
      content: 'Click the sparkles to open your AI research mentor. Ask questions, search papers, and get help.',
      placement: 'right',
      disableBeacon: true,
      spotlightClicks: true, // Allow user to click the element
    },
    {
      target: '[data-tour-id="view-notebook"]',
      content: 'Switch to the Notebook view to write your paper. It supports Markdown, rich text, and exports.',
      placement: 'right',
      spotlightClicks: true,
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
    {
      target: '[data-tour-id="help-trigger"]',
      content: "Need a refresher? Click the '?' button here anytime to take this tour again.",
      placement: 'top',
    },
  ];

  if (!isMounted) return null;

  return (
    <Joyride
      steps={steps}
      run={run}
      stepIndex={stepIndex}
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
