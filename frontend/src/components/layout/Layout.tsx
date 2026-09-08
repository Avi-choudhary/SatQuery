import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutGrid, MessageSquare, Map as MapIcon, LogOut, User, Settings } from 'lucide-react';
import { Panel } from '../ui/Panel';

const Layout = () => {
  const location = useLocation();

  const navItems = [
    { id: 'console', icon: <LayoutGrid size={20} />, label: 'Console', path: '/console' },
    { id: 'map', icon: <MapIcon size={20} />, label: 'Map View', path: '/map' },
    { id: 'history', icon: <MessageSquare size={20} />, label: 'History', path: '/history' },
  ];

  const profileItems = [
    { id: 'profile', icon: <User size={20} />, label: 'Profile', path: '/profile' },
    { id: 'settings', icon: <Settings size={20} />, label: 'Settings', path: '/settings' },
  ];

  return (
    <div className="flex h-screen bg-space-black text-white overflow-hidden">
      {/* Sidebar */}
      <Panel
        variant="glass"
        className="w-64 border-r border-white/10 flex flex-col p-0 rounded-none"
      >
        <div className="p-6">
          <h2 className="text-xl font-bold text-accent-cyan flex items-center gap-2">
            <span className="w-2 h-2 bg-accent-cyan rounded-full animate-pulse" />
            SatQuery
          </h2>
        </div>

        <nav className="flex-1 px-4 space-y-2">
          <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-4 px-4">Workstations</div>
          {navItems.map((item) => (
            <Link
              key={item.id}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                location.pathname === item.path
                  ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="px-4 space-y-2 mb-6">
          <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-4 px-4">Account</div>
          {profileItems.map((item) => (
            <Link
              key={item.id}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                location.pathname === item.path
                  ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </div>

        <div className="p-4 border-t border-white/10">
          <Link
            to="/"
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 hover:text-red-400 hover:bg-red-400/10 transition-all"
          >
            <LogOut size={20} />
            Logout
          </Link>
        </div>
      </Panel>

      {/* Main Content */}
      <main className="flex-1 relative overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
};

export default Layout;
