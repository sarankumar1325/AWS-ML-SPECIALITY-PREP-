import React, { useState } from 'react';
import { QuizMode } from './types';
import QuizEngine from './components/QuizEngine';
import { Cloud, GraduationCap, Timer as TimerIcon, BookOpen } from 'lucide-react';

const App: React.FC = () => {
  const [mode, setMode] = useState<QuizMode>('home');

  const startPractice = () => setMode('practice');
  const startExam = () => setMode('exam');
  const goHome = () => setMode('home');

  return (
    <div className="min-h-screen bg-[#f3f4f6] text-gray-900 font-sans">
      {/* Header */}
      <header className="bg-aws-navy text-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={goHome}>
            <Cloud className="w-8 h-8 text-aws-orange" />
            <span className="text-xl font-bold tracking-tight">AWS ML <span className="text-aws-orange">Specialty</span></span>
          </div>

        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {mode === 'home' ? (
          <div className="animate-in fade-in duration-500">
            <div className="text-center mb-12 mt-8">
              <h1 className="text-4xl md:text-5xl font-extrabold text-aws-navy mb-4">
                Master Your AWS Certification
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Realistic exam simulations and detailed practice modes designed to help you pass with confidence.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
              {/* Practice Mode Card */}
              <div 
                onClick={startPractice}
                className="group relative bg-white rounded-2xl p-8 shadow-sm hover:shadow-xl border border-gray-200 transition-all cursor-pointer overflow-hidden"
              >
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <BookOpen className="w-32 h-32 text-aws-blue" />
                </div>
                <div className="relative z-10">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-6">
                    <BookOpen className="w-6 h-6 text-aws-blue" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-3 group-hover:text-aws-blue transition-colors">Practice Mode</h2>
                  <p className="text-gray-600 mb-6 leading-relaxed">
                    Learn as you go. Get immediate feedback, detailed explanations, and unlimited time per question. Perfect for study sessions.
                  </p>
                  <ul className="space-y-2 mb-8 text-sm text-gray-500">
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/> Immediate Feedback</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/> Explanations Revealed</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/> No Time Pressure</li>
                  </ul>
                  <button className="w-full py-3 bg-white border-2 border-aws-blue text-aws-blue font-bold rounded-lg group-hover:bg-aws-blue group-hover:text-white transition-all">
                    Start Practice
                  </button>
                </div>
              </div>

              {/* Exam Mode Card */}
              <div 
                onClick={startExam}
                className="group relative bg-white rounded-2xl p-8 shadow-sm hover:shadow-xl border border-gray-200 transition-all cursor-pointer overflow-hidden"
              >
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <TimerIcon className="w-32 h-32 text-aws-orange" />
                </div>
                <div className="relative z-10">
                  <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center mb-6">
                    <TimerIcon className="w-6 h-6 text-aws-orange" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-3 group-hover:text-aws-orange transition-colors">Exam Simulation</h2>
                  <p className="text-gray-600 mb-6 leading-relaxed">
                    Simulate the real testing environment. Strict timer, no immediate feedback, and a comprehensive review at the end.
                  </p>
                  <ul className="space-y-2 mb-8 text-sm text-gray-500">
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-orange-500 rounded-full"/> Strict Timer (130m)</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-orange-500 rounded-full"/> Hidden Results</li>
                    <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-orange-500 rounded-full"/> 72% Passing Score</li>
                  </ul>
                  <button className="w-full py-3 bg-aws-navy text-white font-bold rounded-lg group-hover:bg-aws-orange transition-all shadow-md">
                    Start Exam
                  </button>
                </div>
              </div>
            </div>
            

          </div>
        ) : (
          <QuizEngine mode={mode} onExit={goHome} />
        )}
      </main>
    </div>
  );
};

export default App;
