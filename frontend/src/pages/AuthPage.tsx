import { type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '../components/ui/Button';
import { Panel } from '../components/ui/Panel';
import { Mail, Lock, ArrowRight } from 'lucide-react';

interface AuthProps {
  mode: 'login' | 'signup';
}

const AuthPage = ({ mode }: AuthProps) => {
  const navigate = useNavigate();
  const isLogin = mode === 'login';

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-space-black p-6 relative overflow-hidden">
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent-cyan/5 blur-[120px] rounded-full" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-accent-teal/5 blur-[120px] rounded-full" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <Panel variant="glass" className="p-8 border border-white/10 shadow-2xl">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h1>
            <p className="text-slate-400 text-sm">
              {isLogin ? 'Enter your credentials to access the console' : 'Join the next generation of satellite analysis'}
            </p>
          </div>

          <form className="space-y-5" onSubmit={(e: FormEvent) => { e.preventDefault(); navigate('/console'); }}>
            <div className="space-y-2">
              <label className="text-xs font-mono uppercase tracking-wider text-slate-500 ml-1">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                <input
                  type="email"
                  required
                  className="w-full bg-space-black border border-white/10 rounded-lg py-3 pl-10 pr-4 text-sm text-white placeholder:text-white/20 focus:outline-none focus:ring-1 focus:ring-accent-cyan transition-all"
                  placeholder="name@agency.gov"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-mono uppercase tracking-wider text-slate-500 ml-1">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                <input
                  type="password"
                  required
                  className="w-full bg-space-black border border-white/10 rounded-lg py-3 pl-10 pr-4 text-sm text-white placeholder:text-white/20 focus:outline-none focus:ring-1 focus:ring-accent-cyan transition-all"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <Button
              type="submit"
              className="w-full py-3 font-mono text-sm uppercase tracking-widest mt-4"
            >
              {isLogin ? 'Login' : 'Sign Up'}
              <ArrowRight size={16} className="ml-2" />
            </Button>
          </form>

          <div className="mt-8 text-center">
            <button
              type="button"
              onClick={() => navigate(isLogin ? '/signup' : '/login')}
              className="text-xs text-slate-500 hover:text-accent-cyan transition-colors"
            >
              {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Login"}
            </button>
          </div>
        </Panel>
      </motion.div>
    </div>
  );
};

export default AuthPage;
