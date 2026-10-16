import React from 'react';
import { Clock, ShieldAlert } from 'lucide-react';
import { Card, CardContent } from './Card';
import { Badge } from './Badge';

export function ComingSoon({
  title,
  phaseNumber,
  phaseName,
  description,
  plannedFeatures = [],
}) {
  return (
    <div className="space-y-6 max-w-4xl mx-auto py-8">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-slate-100">{title}</h1>
            <Badge variant="warning" className="uppercase tracking-wider">
              Phase {phaseNumber} &bull; Coming Soon
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">{phaseName}</p>
        </div>
      </div>

      <Card>
        <CardContent className="space-y-6">
          <div className="flex items-start gap-4 p-4 rounded-lg bg-amber-950/20 border border-amber-800/40 text-amber-200">
            <Clock className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
            <div className="text-sm space-y-1">
              <p className="font-semibold text-amber-300">Phase Under Development</p>
              <p className="text-amber-200/90 text-xs">
                To guarantee engineering integrity, all upcoming capabilities in the RAG curriculum are strictly developed in sequential phases without pseudo-code or mocked placeholders.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-200">Educational Scope</h3>
            <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
          </div>

          {plannedFeatures.length > 0 && (
            <div className="space-y-3 pt-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Planned Engineering Milestones
              </h4>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                {plannedFeatures.map((feature, idx) => (
                  <li
                    key={idx}
                    className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900/60 border border-slate-700/60 text-slate-300 text-xs"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
