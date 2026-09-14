export type LanguageCode = 'en' | 'hi' | 'hinglish';

export interface UIContent {
  mainTitle: string;
  subTitle: string;
  subTitleNoDataset?: string;
  chooseImagery: string;
  describeTitle: string;
  describeQ1: string;
  describeQ2: string;
  describeQ3: string;
  locateTitle: string;
  locateQ1: string;
  locateQ2: string;
  locateQ3: string;
  compareTitle: string;
  compareQ1: string;
  compareQ2: string;
  compareQ3: string;
  compareNote: string;
  inputPlaceholder: string;
  enterToSend: string;
  offlineWarning: string;
}

export const translations: Record<'en' | 'hi', UIContent> = {
  en: {
    mainTitle: 'What should I look at?',
    subTitle: 'Ask in plain language. The agentic controller picks the specialist and puts its evidence on the map beside you.',
    subTitleNoDataset: 'Attach a Sentinel scene or pick a demo one, then ask in plain language.',
    chooseImagery: 'Choose imagery',
    describeTitle: 'Describe the scene',
    describeQ1: 'What land cover types dominate this scene?',
    describeQ2: 'Is this area suitable for growing sugarcane?',
    describeQ3: 'Summarise the vegetation health visible here.',
    locateTitle: 'Locate features',
    locateQ1: 'Find the industrial storage tanks in this scene.',
    locateQ2: 'Where are the water bodies?',
    locateQ3: 'Outline the built-up areas.',
    compareTitle: 'Compare over time',
    compareQ1: 'How much built-up area was added between these dates?',
    compareQ2: 'What changed most between T1 and T2?',
    compareQ3: 'Show me where vegetation was lost.',
    compareNote: 'Needs a bi-temporal pair',
    inputPlaceholder: 'Ask about satellite or SAR imagery — or attach a scene first',
    enterToSend: 'Enter to send · Shift+Enter for a new line',
    offlineWarning: 'Backend unreachable — start the FastAPI server to run queries.',
  },
  hi: {
    mainTitle: 'मुझे क्या देखना चाहिए?',
    subTitle: 'सरल भाषा में पूछें। एजेंटिक कंट्रोलर विशेषज्ञ को चुनता है और उसका साक्ष्य आपके सामने मानचित्र पर दिखाता है।',
    subTitleNoDataset: 'एक सेंटिनल सीन संलग्न करें या एक डेमो चुनें, फिर सरल भाषा में पूछें।',
    chooseImagery: 'इमेजरी चुनें',
    describeTitle: 'दृश्य का वर्णन करें',
    describeQ1: 'इस दृश्य में किस प्रकार के भूमि आवरण का दबदबा है?',
    describeQ2: 'क्या यह क्षेत्र गन्ना उगाने के लिए उपयुक्त है?',
    describeQ3: 'यहाँ दिखाई देने वाले वनस्पति स्वास्थ्य का सारांश दें।',
    locateTitle: 'सुविधाएँ खोजें',
    locateQ1: 'इस दृश्य में औद्योगिक भंडारण टैंक खोजें।',
    locateQ2: 'जल निकाय कहाँ हैं?',
    locateQ3: 'निर्मित क्षेत्रों की रूपरेखा तैयार करें।',
    compareTitle: 'समय के साथ तुलना करें',
    compareQ1: 'इन तिथियों के बीच कितना निर्मित क्षेत्र जोड़ा गया?',
    compareQ2: 'T1 और T2 के बीच सबसे ज़्यादा क्या बदला?',
    compareQ3: 'मुझे दिखाएं कि वनस्पति कहाँ नष्ट हुई थी।',
    compareNote: 'द्वि-सामयिक (Bi-temporal) जोड़ी की आवश्यकता है',
    inputPlaceholder: 'उपग्रह या SAR इमेजरी के बारे में पूछें — या पहले एक सीन संलग्न करें',
    enterToSend: 'भेजने के लिए Enter दबाएं · नई लाइन के लिए Shift+Enter दबाएं',
    offlineWarning: 'बैकएंड अनुपलब्ध है — क्वेरी चलाने के लिए FastAPI सर्वर चालू करें।',
  },
};