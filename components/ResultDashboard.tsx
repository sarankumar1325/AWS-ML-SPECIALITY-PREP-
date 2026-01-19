import React from 'react';
import { Question, QuizState } from '../types';
import { PASSING_SCORE_PERCENTAGE } from '../data/questions';
import { CheckCircle2, XCircle, AlertTriangle, RefreshCcw, Home } from 'lucide-react';

interface ResultDashboardProps {
  quizState: QuizState;
  questions: Question[];
  onRestart: () => void;
  onHome: () => void;
}

const ResultDashboard: React.FC<ResultDashboardProps> = ({
  quizState,
  questions,
  onRestart,
  onHome,
}) => {
  const totalQuestions = questions.length;
  const correctQuestions = questions.filter(q => {
    const selected = quizState.selectedAnswers[q.id] || [];
    const correct = q.correctAnswerIds;
    return (
      selected.length === correct.length &&
      selected.every(s => correct.includes(s))
    );
  });

  const correctCount = correctQuestions.length;
  const scorePercentage = Math.round((correctCount / totalQuestions) * 100);
  const isPassed = scorePercentage >= PASSING_SCORE_PERCENTAGE;

  // Identify incorrect questions for review
  const incorrectQuestions = questions.filter(q => {
    const selected = quizState.selectedAnswers[q.id] || [];
    const correct = q.correctAnswerIds;
    return !(
      selected.length === correct.length &&
      selected.every(s => correct.includes(s))
    );
  });

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      {/* Score Header */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-8 text-center">
        <h2 className="text-2xl font-bold text-gray-700 mb-6">Exam Results</h2>
        
        <div className="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-16">
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Simple SVG Circle Chart */}
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="96"
                cy="96"
                r="88"
                stroke="#e5e7eb"
                strokeWidth="12"
                fill="none"
              />
              <circle
                cx="96"
                cy="96"
                r="88"
                stroke={isPassed ? "#22c55e" : "#ef4444"}
                strokeWidth="12"
                fill="none"
                strokeDasharray={2 * Math.PI * 88}
                strokeDashoffset={2 * Math.PI * 88 * (1 - scorePercentage / 100)}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-4xl font-black text-gray-800">{scorePercentage}%</span>
              <span className={`text-sm font-bold uppercase tracking-wider ${isPassed ? 'text-green-600' : 'text-red-500'}`}>
                {isPassed ? 'Passed' : 'Failed'}
              </span>
            </div>
          </div>

          <div className="text-left space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle2 className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Correct Answers</p>
                <p className="text-xl font-bold text-gray-800">{correctCount} <span className="text-gray-400 text-sm font-normal">/ {totalQuestions}</span></p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="p-2 bg-red-100 rounded-lg">
                <XCircle className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Incorrect Answers</p>
                <p className="text-xl font-bold text-gray-800">{totalQuestions - correctCount}</p>
              </div>
            </div>

            <div className="pt-4 flex gap-3">
              <button onClick={onRestart} className="flex items-center gap-2 px-4 py-2 bg-aws-navy text-white rounded-lg hover:bg-aws-squid transition-colors">
                <RefreshCcw className="w-4 h-4" /> Try Again
              </button>
              <button onClick={onHome} className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
                <Home className="w-4 h-4" /> Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Incorrect Questions Review */}
      {incorrectQuestions.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="bg-red-50 px-6 py-4 border-b border-red-100 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <h3 className="font-bold text-red-800">Incorrect Questions Review</h3>
          </div>
          
          <div className="divide-y divide-gray-100">
            {incorrectQuestions.map((q, idx) => {
              const userSelected = quizState.selectedAnswers[q.id] || [];
              
              return (
                <div key={q.id} className="p-6 md:p-8 hover:bg-gray-50 transition-colors">
                  <div className="mb-4">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Question {questions.findIndex(allQ => allQ.id === q.id) + 1}</span>
                    <h4 className="text-lg font-medium text-gray-900 mt-1">{q.text}</h4>
                  </div>

                  <div className="space-y-2 mb-6">
                    {q.options.map(opt => {
                      const isSelected = userSelected.includes(opt.id);
                      const isCorrect = q.correctAnswerIds.includes(opt.id);
                      
                      let style = "border-gray-200 text-gray-500 opacity-60";
                      let icon = null;

                      if (isCorrect) {
                        style = "border-green-200 bg-green-50 text-green-800 opacity-100 ring-1 ring-green-500";
                        icon = <CheckCircle2 className="w-4 h-4 text-green-600" />;
                      } else if (isSelected) {
                        style = "border-red-200 bg-red-50 text-red-800 opacity-100 ring-1 ring-red-500";
                        icon = <XCircle className="w-4 h-4 text-red-600" />;
                      }

                      return (
                        <div key={opt.id} className={`flex items-center justify-between p-3 border rounded-lg text-sm ${style}`}>
                          <span>{opt.text}</span>
                          {icon}
                        </div>
                      );
                    })}
                  </div>

                  <div className="bg-blue-50 p-4 rounded-lg text-sm text-blue-900 border border-blue-100">
                    <span className="font-bold block mb-1">Explanation:</span>
                    {q.explanation}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDashboard;
