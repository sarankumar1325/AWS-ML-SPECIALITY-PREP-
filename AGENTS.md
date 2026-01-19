# AGENTS.md

## Build Commands

This project uses Bun as the package manager with Vite as the build tool.

```bash
# Install dependencies (if needed)
bun install

# Start development server (runs on port 3000)
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview

# TypeScript type checking
bunx tsc --noEmit
```

**Note:** This project does not currently have a test suite. When adding tests, create a `*.test.tsx` or `*.spec.tsx` file and configure a test framework (Jest, Vitest, etc.).

## Project Structure

```
/
├── App.tsx              # Main application component
├── index.tsx            # Entry point
├── types.ts             # TypeScript type definitions
├── data/
│   └── questions.ts     # Quiz questions data
├── components/          # React components
│   ├── QuestionCard.tsx
│   ├── QuizEngine.tsx
│   ├── ResultDashboard.tsx
│   └── Timer.tsx
```

## Code Style Guidelines

### TypeScript & React

- Use functional components with `React.FC` type annotation
- Export components using named exports
- Define interfaces for component props with `Props` suffix
- Use TypeScript for all types - no `any` types allowed

```tsx
interface ComponentNameProps {
  prop1: string;
  prop2: number;
  onAction: () => void;
}

const ComponentName: React.FC<ComponentNameProps> = ({ prop1, prop2, onAction }) => {
  // component logic
  return <div>{/* JSX */}</div>;
};

export default ComponentName;
```

### Imports

Import order (grouped, separated by blank lines):
1. React imports
2. Third-party dependencies (lucide-react, clsx, etc.)
3. Local imports (prefixed with `@/` for root directory)

```tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Icon1, Icon2 } from 'lucide-react';
import { clsx } from 'clsx';

import { Type1, Type2 } from '@/types';
import { Component1 } from '@/components/Component1';
import { data, CONSTANT } from '@/data/some-data';
```

### State Management

- Use `useState` for local state
- Use `useCallback` for event handlers passed as props
- Use functional updates for derived state: `setState(prev => ...)`
- Destructure state updates for readability

```tsx
const [count, setCount] = useState(0);

const handleClick = useCallback(() => {
  setCount(prev => prev + 1);
}, []);

const handleComplexUpdate = () => {
  setState(prev => ({
    ...prev,
    field: newValue,
  }));
};
```

### Styling

- Use Tailwind CSS utility classes
- Use `clsx` for conditional class names
- Define AWS color theme using custom variables: `text-aws-navy`, `bg-aws-orange`, etc.
- Follow mobile-first responsive design with `md:` breakpoints

```tsx
<div className={clsx(
  "base-classes",
  condition && "conditional-classes",
  isActive && "active:classes"
)}>
```

### Error Handling

- Throw descriptive errors for critical failures (e.g., missing DOM elements)
- Check for null/undefined before accessing properties
- Use TypeScript optional chaining `?.` and nullish coalescing `??`

```tsx
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}
```

### Naming Conventions

- **Components:** PascalCase (`QuestionCard`)
- **Files:** PascalCase.tsx for components, camelCase.ts for utilities/data
- **Variables/Functions:** camelCase (`handleClick`, `questionData`)
- **Constants:** UPPER_SNAKE_CASE (`EXAM_DURATION_MINUTES`)
- **Interfaces/Types:** PascalCase with descriptive names (`Question`, `QuizState`)
- **Props interfaces:** `${ComponentName}Props` pattern

### Data Structure

- Export constants and data from `data/questions.ts`
- Use string IDs for entities (`id: 'q1'`)
- Type arrays explicitly: `questions: Question[]`

### Component Patterns

- Keep components focused and single-purpose
- Extract complex logic into helper functions or custom hooks
- Use `key` prop for list iterations
- Close tags explicitly: `<div></div>` rather than `<div />` when content exists

### TypeScript Configuration

- Target: ES2022
- Module resolution: bundler
- Path alias: `@/*` maps to root directory
- JSX transform: `react-jsx` (no React import needed for JSX)
