import Hero from '../components/features/Hero';
import HowItWorks from '../components/features/HowItWorks';
import Capabilities from '../components/features/Capabilities';
import Console from '../components/features/Console';
import TechStack from '../components/features/TechStack';
import Footer from '../components/features/Footer';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-space-black text-white selection:bg-accent-cyan selection:text-space-black">
      <Hero />
      <HowItWorks />
      <Console />
      <Capabilities />
      <TechStack />
      <Footer />
    </div>
  );
};

export default LandingPage;
