export type QuestionType = 'MCQ' | 'MSQ';

export interface Option {
  id: string;
  text: string;
}

export interface Question {
  id: string;
  text: string;
  type: QuestionType;
  options: Option[];
  correctAnswerIds: string[];
  explanation: string;
  domain: string;
}

export type QuizMode = 'home' | 'practice' | 'exam';

export interface QuizState {
  mode: QuizMode;
  currentQuestionIndex: number;
  selectedAnswers: Record<string, string[]>; // questionId -> array of optionIds
  isSubmitted: boolean;
  score: number;
  startTime: number | null;
  endTime: number | null;
  timeRemaining: number; // in seconds
}
