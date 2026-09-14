import React, { useState, useRef, useEffect } from 'react';
import { Languages } from 'lucide-react';
import { IconButton } from './Button';
import { useLanguage } from '../../context/LanguageContext';

export const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useLanguage();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={ref} className="relative inline-block">
      <IconButton
        label="Change language"
        size="sm"
        active={open}
        onClick={() => setOpen(!open)}
      >
        <Languages size={13} />
      </IconButton>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1.5 min-w-[110px] overflow-hidden rounded-xl border border-line bg-surface p-1 shadow-xl">
          <button
            type="button"
            onClick={() => {
              setLanguage('en');
              setOpen(false);
            }}
            className={`flex w-full items-center px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer ${
              language === 'en'
                ? 'bg-accent/15 text-accent font-medium'
                : 'text-ink-muted hover:bg-surface-3 hover:text-ink'
            }`}
          >
            English
          </button>
          <button
            type="button"
            onClick={() => {
              setLanguage('hi');
              setOpen(false);
            }}
            className={`flex w-full items-center px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer ${
              language === 'hi'
                ? 'bg-accent/15 text-accent font-medium'
                : 'text-ink-muted hover:bg-surface-3 hover:text-ink'
            }`}
          >
            हिंदी
          </button>
        </div>
      )}
    </div>
  );
};

export default LanguageSelector;