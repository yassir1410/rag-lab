import React from 'react';

export function Badge({ children, variant = 'default', className = '' }) {
  const variantStyles = {
    default: 'bg-slate-800 text-slate-300 border-slate-700',
    primary: 'bg-blue-900/60 text-blue-300 border-blue-700/60',
    success: 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60',
    warning: 'bg-amber-950/80 text-amber-300 border-amber-700/60',
    danger: 'bg-rose-950/80 text-rose-300 border-rose-700/60',
    secondary: 'bg-slate-700 text-slate-200 border-slate-600',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
        variantStyles[variant] || variantStyles.default
      } ${className}`}
    >
      {children}
    </span>
  );
}
