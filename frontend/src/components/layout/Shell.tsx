import { useState, useEffect, useRef } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/Button';

export function Shell() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const hamburgerRef = useRef<HTMLButtonElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const drawerRef = useRef<HTMLDivElement | null>(null);
  const wasOpenRef = useRef(false);

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Focus management: when drawer opens, focus close button; when it closes, return focus to hamburger
  useEffect(() => {
    if (mobileMenuOpen) {
      wasOpenRef.current = true;
      // Focus the close button once drawer opens
      const timer = setTimeout(() => {
        closeButtonRef.current?.focus();
      }, 50);
      return () => clearTimeout(timer);
    } else if (wasOpenRef.current) {
      wasOpenRef.current = false;
      hamburgerRef.current?.focus();
    }
  }, [mobileMenuOpen]);

  // Handle keyboard: Escape closes drawer, Tab traps focus within drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!mobileMenuOpen) return;

      if (e.key === 'Escape') {
        e.preventDefault();
        setMobileMenuOpen(false);
        return;
      }

      if (e.key === 'Tab' && drawerRef.current) {
        const focusable = drawerRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;

        const firstElement = focusable[0];
        const lastElement = focusable[focusable.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileMenuOpen]);

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Desktop Persistent Sidebar */}
      <div className="hidden lg:flex shrink-0">
        <Sidebar />
      </div>

      {/* Mobile Backdrop & Drawer */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs lg:hidden animate-in fade-in-0 duration-200"
          onClick={() => setMobileMenuOpen(false)}
          aria-hidden="true"
        />
      )}
      <div
        ref={drawerRef}
        id="mobile-navigation-drawer"
        className={`fixed inset-y-0 left-0 z-50 w-72 bg-surface transform transition-transform duration-200 ease-in-out lg:hidden flex flex-col shadow-2xl ${
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="Mobile Navigation"
      >
        <div className="absolute top-3 right-3 z-10">
          <Button
            ref={closeButtonRef}
            variant="ghost"
            size="icon"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close menu"
            className="text-text-muted hover:text-text-primary"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>
        <Sidebar onNavClick={() => setMobileMenuOpen(false)} className="w-full border-r-0" />
      </div>

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header
          onMobileMenuToggle={() => setMobileMenuOpen((prev) => !prev)}
          isMobileMenuOpen={mobileMenuOpen}
          hamburgerRef={hamburgerRef}
        />
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6">
          <div className="max-w-7xl mx-auto w-full">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
