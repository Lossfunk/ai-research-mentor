"use client";

import { useState } from "react";
import { ArrowRight, BookOpen, Feather } from "lucide-react";

export default function LandingPage({ onEnter }: { onEnter: () => void }) {
  const [isExiting, setIsExiting] = useState(false);

  const handleEnter = () => {
    setIsExiting(true);
    // Wait for exit animation to finish before triggering parent state change
    setTimeout(() => {
      onEnter();
    }, 400); 
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEnter();
    }
  };

  return (
    <main className={`h-screen w-screen flex items-center justify-center bg-[#F7F6F3] transition-all duration-500 ease-in-out ${isExiting ? 'opacity-0 scale-95 filter blur-sm' : 'opacity-100 scale-100'}`}>
      {/* Background Grain Texture (CSS Pattern would go here, using simple opacity for now) */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")` }} />

      <div className="relative z-10 max-w-2xl w-full px-6 flex flex-col items-center text-center space-y-8">
        
        {/* Logo / Icon */}
        <div className="animate-slide-down" style={{ animationDelay: '0.1s' }}>
          <div className="w-16 h-16 bg-stone-900 text-white rounded-2xl flex items-center justify-center shadow-xl mb-6 mx-auto">
            <Feather size={32} strokeWidth={1.5} />
          </div>
        </div>

        {/* Typography */}
        <div className="space-y-4 animate-slide-up" style={{ animationDelay: '0.2s' }}>
          <h1 className="font-serif text-5xl md:text-7xl font-medium text-stone-900 tracking-tight">
            Academic <span className="italic text-stone-600">Research</span> Mentor
          </h1>
          <p className="text-lg md:text-xl text-stone-500 font-sans max-w-md mx-auto leading-relaxed">
            Your focused studio for intelligent discovery, synthesis, and writing.
          </p>
        </div>

        {/* Interaction Area */}
        <div className="w-full max-w-md animate-slide-up pt-8" style={{ animationDelay: '0.4s' }}>
          <div className="group relative">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-stone-200 to-stone-300 rounded-xl opacity-50 group-hover:opacity-100 transition duration-200 blur"></div>
            <button 
              onClick={handleEnter}
              className="relative w-full bg-white hover:bg-stone-50 text-stone-800 font-serif text-xl py-4 px-6 rounded-xl shadow-sm border border-stone-100 flex items-center justify-between transition-all duration-200 active:scale-[0.98]"
            >
              <span className="pl-2">Enter Studio</span>
              <div className="bg-stone-100 p-2 rounded-lg text-stone-400 group-hover:text-stone-900 transition-colors">
                <ArrowRight size={20} />
              </div>
            </button>
          </div>
          <div className="mt-4 flex items-center justify-center gap-6 text-sm text-stone-400 font-sans">
            <span>Vol. 1.0</span>
            <span className="w-1 h-1 bg-stone-300 rounded-full"></span>
            <span>Established 2025</span>
          </div>
        </div>

      </div>
    </main>
  );
}
