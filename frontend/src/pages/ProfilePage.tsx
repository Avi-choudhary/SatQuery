import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';

const ProfilePage = () => {
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-8">User Profile</h1>

      <div className="grid gap-6">
        <Panel variant="glass" className="p-6">
          <div className="flex items-center gap-6">
            <div className="w-24 h-24 rounded-full bg-accent-cyan/20 border border-accent-cyan/30 flex items-center justify-center text-accent-cyan text-3xl font-bold">
              JD
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">John Doe</h2>
              <p className="text-sm text-slate-400">Senior Analyst, ISRO</p>
            </div>
          </div>
        </Panel>

        <Panel variant="glass" className="p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Account Details</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center border-b border-white/5 pb-2">
              <span className="text-sm text-slate-500">Email</span>
              <span className="text-sm text-white">john.doe@isro.gov.in</span>
            </div>
            <div className="flex justify-between items-center border-b border-white/5 pb-2">
              <span className="text-sm text-slate-500">Role</span>
              <span className="text-sm text-accent-cyan font-mono">ADMIN</span>
            </div>
          </div>
          <div className="mt-6">
            <Button variant="outline" size="sm">Update Profile</Button>
          </div>
        </Panel>
      </div>
    </div>
  );
};

export default ProfilePage;
