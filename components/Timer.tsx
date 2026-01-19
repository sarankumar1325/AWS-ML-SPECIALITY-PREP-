import React, { useEffect } from 'react';
import { Clock } from 'lucide-react';
import { clsx } from 'clsx';

interface TimerProps {
  timeRemaining: number;
  setTimeRemaining: React.Dispatch<React.SetStateAction<number>>;
  onTimeUp: () => void;
  isActive: boolean;
}

const Timer: React.FC<TimerProps> = ({ timeRemaining, setTimeRemaining, onTimeUp, isActive }) => {
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            clearInterval(interval);
            onTimeUp();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isActive, timeRemaining, setTimeRemaining, onTimeUp]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const isLowTime = timeRemaining < 300; // Less than 5 minutes

  return (
    <div className={clsx(
      "flex items-center space-x-2 px-4 py-2 rounded-lg font-mono font-bold transition-colors duration-300 border",
      isLowTime 
        ? "bg-red-50 text-red-600 border-red-200 animate-pulse" 
        : "bg-white text-aws-navy border-gray-200"
    )}>
      <Clock className="w-5 h-5" />
      <span className="text-xl tracking-widest">{formatTime(timeRemaining)}</span>
      {isLowTime && <span className="text-xs uppercase font-bold ml-2">Warning</span>}
    </div>
  );
};

export default Timer;
