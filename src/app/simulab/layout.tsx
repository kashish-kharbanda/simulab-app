'use client';

import { useSearchParams } from 'next/navigation';
import React, { useEffect, Suspense } from 'react';

function SearchParamsHandler() {
  const search = useSearchParams();

  // Keep task_id stable across tab nav via querystring and localStorage
  useEffect(() => {
    const qTask = search.get('task_id');
    if (qTask) {
      try {
        localStorage.setItem('simulab_current_task_id', qTask);
      } catch {}
    }
  }, [search]);

  return null;
}

export default function SimulabSectionLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full h-full">
      <Suspense fallback={null}>
        <SearchParamsHandler />
      </Suspense>
      
      {/* Route header with product title and subtitle */}
      <div className="w-full border-b border-gray-100 bg-white">
        <div className="max-w-[1200px] mx-auto px-4 py-4 flex flex-col items-center text-center">
          <div className="text-2xl font-semibold tracking-tight text-slate-900">
            SimuLab
          </div>
          <div className="text-sm text-slate-500 mt-1">
            Accelerate Lead Molecule Discovery and De-Risk R&D
          </div>
        </div>
      </div>
      <div className="w-full">
        <div className="max-w-[1200px] mx-auto px-4 py-4">
          {children}
        </div>
      </div>
    </div>
  );
}


