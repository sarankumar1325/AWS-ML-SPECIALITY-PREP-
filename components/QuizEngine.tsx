import React, { useState, useCallback } from 'react';
import { questions, EXAM_DURATION_MINUTES } from '../data/questions';
import { QuizMode, QuizState } from '../types';
import QuestionCard from './QuestionCard';
import ResultDashboard from './ResultDashboard';
import Timer from './Timer';
import { ChevronRight, ChevronLeft, Send, LayoutGrid } from 'lucide-react';

interface QuizEngineProps {
  mode: QuizMode;
  onExit: () => void;
}

const QuizEngine: React.FC<QuizEngineProps> = ({ mode, onExit }) => {
  const [state, setState] = useState<QuizState>({
    mode,
    currentQuestionIndex: 0,
    selectedAnswers: {},
    isSubmitted: false,
    score: 0,
    startTime: Date.now(),
    endTime: null,
    timeRemaining: EXAM_DURATION_MINUTES * 60,
  });

  const currentQuestion = questions[state.currentQuestionIndex];
  const isLastQuestion = state.currentQuestionIndex === questions.length - 1;

  // Handle Answer Selection
  const handleAnswerSelect = (optionId: string) => {
    if (state.isSubmitted) return;

    setState((prev) => {
      const currentSelected = prev.selectedAnswers[currentQuestion.id] || [];
      let newSelected: string[];

      if (currentQuestion.type === 'MCQ') {
        // Radio button logic: replace entire selection
        newSelected = [optionId];
      } else {
        // Checkbox logic: toggle selection
        if (currentSelected.includes(optionId)) {
          newSelected = currentSelected.filter((id) => id !== optionId);
        } else {
          // MSQ constraint: In strict exam mode, usually AWS allows selecting more or less, 
          // but visually we allow free selection. The user must enforce the count themselves 
          // usually, but providing a visual limit is good UX.
          // For this app, we allow toggling freely.
          newSelected = [...currentSelected, optionId];
        }
      }

      return {
        ...prev,
        selectedAnswers: {
          ...prev.selectedAnswers,
          [currentQuestion.id]: newSelected,
        },
      };
    });
  };

  const handleNext = () => {
    if (state.currentQuestionIndex < questions.length - 1) {
      setState((prev) => ({ ...prev, currentQuestionIndex: prev.currentQuestionIndex + 1 }));
    }
  };

  const handlePrev = () => {
    if (state.currentQuestionIndex > 0) {
      setState((prev) => ({ ...prev, currentQuestionIndex: prev.currentQuestionIndex - 1 }));
    }
  };

  const handleSubmit = useCallback(() => {
    setState((prev) => ({
      ...prev,
      isSubmitted: true,
      endTime: Date.now(),
    }));
  }, []);

  const handleRestart = () => {
    setState({
      mode,
      currentQuestionIndex: 0,
      selectedAnswers: {},
      isSubmitted: false,
      score: 0,
      startTime: Date.now(),
      endTime: null,
      timeRemaining: EXAM_DURATION_MINUTES * 60,
    });
  };

  // Render Results if submitted
  if (state.isSubmitted) {
    return (
      <ResultDashboard 
        quizState={state} 
        questions={questions} 
        onRestart={handleRestart}
        onHome={onExit}
      />
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Top Bar: Timer & Progress */}
      <div className="sticky top-0 z-10 bg-[#f3f4f6]/95 backdrop-blur-sm pt-4 pb-4 mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
            <button onClick={onExit} className="text-sm font-medium text-gray-500 hover:text-aws-navy flex items-center gap-1">
                <LayoutGrid className="w-4 h-4" /> Exit
            </button>
            {mode === 'exam' && (
            <Timer 
                timeRemaining={state.timeRemaining} 
                setTimeRemaining={(val) => setState(prev => ({ ...prev, timeRemaining: typeof val === 'function' ? val(prev.timeRemaining) : val }))}
                onTimeUp={handleSubmit}
                isActive={!state.isSubmitted}
            />
            )}
        </div>

        <div className="hidden md:flex flex-col items-end">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">Progress</span>
            <div className="flex items-center gap-2">
                <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                        className="h-full bg-aws-orange transition-all duration-300"
                        style={{ width: `${((state.currentQuestionIndex + 1) / questions.length) * 100}%` }}
                    />
                </div>
                <span className="text-sm font-bold text-aws-navy">{state.currentQuestionIndex + 1}/{questions.length}</span>
            </div>
        </div>
      </div>

      {/* Main Question Area */}
      <QuestionCard 
        question={currentQuestion}
        selectedAnswers={state.selectedAnswers[currentQuestion.id] || []}
        onAnswerSelect={handleAnswerSelect}
        mode={mode}
        questionNumber={state.currentQuestionIndex + 1}
        totalQuestions={questions.length}
      />

      {/* Footer Navigation */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 md:static md:bg-transparent md:border-0 md:p-0 md:mt-8">
        <div className="max-w-5xl mx-auto flex justify-between items-center">
            <button
                onClick={handlePrev}
                disabled={state.currentQuestionIndex === 0}
                className="flex items-center px-4 py-2 rounded-lg text-gray-600 bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-all"
            >
                <ChevronLeft className="w-5 h-5 mr-1" /> Previous
            </button>

            {isLastQuestion ? (
                <button
                    onClick={handleSubmit}
                    className="flex items-center px-6 py-2 rounded-lg text-white bg-aws-orange hover:bg-yellow-500 shadow-md transition-all font-bold"
                >
                    Submit Exam <Send className="w-4 h-4 ml-2" />
                </button>
            ) : (
                <button
                    onClick={handleNext}
                    className="flex items-center px-6 py-2 rounded-lg text-white bg-aws-navy hover:bg-aws-squid shadow-md transition-all"
                >
                    Next <ChevronRight className="w-5 h-5 ml-1" />
                </button>
            )}
        </div>
      </div>
      
      {/* Spacer for mobile fixed footer */}
      <div className="h-20 md:hidden" />
    </div>
  );
};

export default QuizEngine;
