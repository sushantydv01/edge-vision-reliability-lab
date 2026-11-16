import { useLocation } from 'react-router-dom';
import { StatusIndicator } from './StatusIndicator';
import { NAV_ITEMS } from './Sidebar';
import { Menu, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface HeaderProps {
  onMobileMenuToggle: () => void;
}

export function Header({ onMobileMenuToggle }: HeaderProps) {
  const location = useLocation();
  const currentNav = NAV_ITEMS.find((item) =>
    item.path === '/'
      ? location.pathname === '/'
      : location.pathname.startsWith(item.path)
  ) || { name: 'Dashboard' };

  return (
    <header className="h-14 border-b border-surface-border bg-surface/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between sticky top-0 z-30 select-none">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onMobileMenuToggle}
          className="lg:hidden text-text-secondary hover:text-text-primary"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </Button>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-muted hidden sm:inline">Lab /</span>
          <span className="font-semibold text-text-primary tracking-tight">
            {currentNav.name}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Device Chip */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface-secondary border border-surface-border text-xs text-text-secondary">
          <Cpu className="w-3.5 h-3.5 text-accent" />
          <span className="font-mono text-[11px] text-text-primary">MPS / CPU</span>
        </div>

        {/* Backend Connectivity Status */}
        <StatusIndicator />
      </div>
    </header>
  );
}
