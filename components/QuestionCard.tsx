import React, { useState } from 'react';
import { Question, QuizMode } from '../types';
import { CheckCircle2, XCircle, AlertCircle, HelpCircle } from 'lucide-react';
import { clsx } from 'clsx';

interface QuestionCardProps {
  question: Question;
  selectedAnswers: string[];
  onAnswerSelect: (optionId: string) => void;
  mode: QuizMode;
  questionNumber: number;
  totalQuestions: number;
}

const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  selectedAnswers,
  onAnswerSelect,
  mode,
  questionNumber,
  totalQuestions,
}) => {
  const [showPracticeFeedback, setShowPracticeFeedback] = useState(false);
  
  // Reset local practice feedback state when question changes
  React.useEffect(() => {
    setShowPracticeFeedback(false);
  }, [question.id]);

  const isSelected = (id: string) => selectedAnswers.includes(id);

  const handleOptionClick = (optionId: string) => {
    if (showPracticeFeedback && mode === 'practice') return; // Lock after checking
    onAnswerSelect(optionId);
  };

  const checkPracticeAnswer = () => {
    setShowPracticeFeedback(true);
  };

  const isCorrect = (optionId: string) => question.correctAnswerIds.includes(optionId);

  // Helper to determine styling for an option in Practice Mode
  const getOptionStyle = (optionId: string) => {
    const selected = isSelected(optionId);
    
    // Exam Mode: Just show selected state, no correctness feedback
    if (mode === 'exam') {
      return selected
        ? 'border-aws-orange bg-orange-50 ring-1 ring-aws-orange'
        : 'border-gray-200 hover:border-aws-orange/50 hover:bg-gray-50';
    }

    // Practice Mode: Show selected state initially, then green/red after "Check"
    if (!showPracticeFeedback) {
      return selected
        ? 'border-aws-blue bg-blue-50 ring-1 ring-aws-blue'
        : 'border-gray-200 hover:border-aws-blue/50 hover:bg-gray-50';
    }

    // Practice Mode (Feedback Shown)
    if (isCorrect(optionId)) {
      return 'border-green-500 bg-green-50 ring-1 ring-green-500';
    }
    if (selected && !isCorrect(optionId)) {
      return 'border-red-500 bg-red-50 ring-1 ring-red-500';
    }
    return 'border-gray-200 opacity-60';
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <span className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
          Question {questionNumber} of {totalQuestions}
        </span>
        <span className="text-xs font-medium px-2 py-1 bg-gray-200 text-gray-600 rounded">
          {question.domain}
        </span>
      </div>

      <div className="p-6 md:p-8">
        {/* Question Text */}
        <h2 className="text-xl font-medium text-aws-navy mb-6 leading-relaxed">
          {question.text}
        </h2>

        {/* Instructions */}
        <div className="mb-4 flex items-center gap-2 text-sm text-gray-500 italic">
          <HelpCircle className="w-4 h-4" />
          {question.type === 'MCQ' 
            ? 'Select one option.' 
            : `Select ${question.correctAnswerIds.length} options.`}
        </div>

        {/* Options */}
        <div className="space-y-3">
          {question.options.map((option) => (
            <div
              key={option.id}
              onClick={() => handleOptionClick(option.id)}
              className={clsx(
                "group relative flex items-start p-4 cursor-pointer border rounded-lg transition-all duration-200",
                getOptionStyle(option.id)
              )}
            >
              <div className="flex items-center h-5">
                {question.type === 'MCQ' ? (
                  <div className={clsx(
                    "w-5 h-5 rounded-full border flex items-center justify-center",
                    isSelected(option.id) 
                      ? (mode === 'practice' && showPracticeFeedback 
                          ? (isCorrect(option.id) ? "border-green-500" : "border-red-500") 
                          : "border-aws-orange bg-aws-orange")
                      : "border-gray-300 group-hover:border-gray-400"
                  )}>
                    {isSelected(option.id) && <div className="w-2 h-2 bg-white rounded-full" />}
                  </div>
                ) : (
                  <div className={clsx(
                    "w-5 h-5 rounded border flex items-center justify-center",
                    isSelected(option.id)
                      ? (mode === 'practice' && showPracticeFeedback
                          ? (isCorrect(option.id) ? "border-green-500 bg-green-500" : "border-red-500 bg-red-500")
                          : "border-aws-orange bg-aws-orange")
                      : "border-gray-300 group-hover:border-gray-400"
                  )}>
                    {isSelected(option.id) && (
                      <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                )}
              </div>
              <div className="ml-3 text-sm md:text-base text-gray-800">
                {option.text}
              </div>
              
              {/* Status Icons for Practice Mode Feedback */}
              {mode === 'practice' && showPracticeFeedback && (
                <div className="ml-auto pl-4">
                  {isCorrect(option.id) && <CheckCircle2 className="w-5 h-5 text-green-500" />}
                  {isSelected(option.id) && !isCorrect(option.id) && <XCircle className="w-5 h-5 text-red-500" />}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Practice Mode: Check Answer Button */}
        {mode === 'practice' && !showPracticeFeedback && (
          <div className="mt-8">
            <button
              onClick={checkPracticeAnswer}
              disabled={selectedAnswers.length === 0}
              className="px-6 py-2 bg-aws-navy text-white rounded-md font-medium hover:bg-aws-squid disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Check Answer
            </button>
          </div>
        )}

        {/* Practice Mode: Explanation */}
        {mode === 'practice' && showPracticeFeedback && (
          <div className="mt-8 p-6 bg-blue-50 border-l-4 border-aws-blue rounded-r-lg animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-lg font-bold text-aws-navy flex items-center gap-2 mb-2">
              <AlertCircle className="w-5 h-5 text-aws-blue" />
              Explanation
            </h3>
            <p className="text-gray-700 leading-relaxed">
              {question.explanation}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuestionCard;
