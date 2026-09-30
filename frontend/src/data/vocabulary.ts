export type VocabularyCategory =
  | "Greetings & Social"
  | "Basic Needs"
  | "Food & Drink"
  | "Family & People"
  | "Time & Calendar"
  | "Places & Travel"
  | "Feelings & Emergency"
  | "Common Verbs & Questions"
  | "Daily Life"
  | "Accessibility";

export type VocabularyItem = {
  id: string;
  phrase: string;
  category: VocabularyCategory;
  custom?: boolean;
};

export const VOCABULARY_CATEGORIES: VocabularyCategory[] = [
  "Greetings & Social",
  "Basic Needs",
  "Food & Drink",
  "Family & People",
  "Time & Calendar",
  "Places & Travel",
  "Feelings & Emergency",
  "Common Verbs & Questions",
  "Daily Life",
  "Accessibility",
];

const rows: Array<[VocabularyCategory, string[]]> = [
  ["Greetings & Social", [
    "Hello", "Hi", "Good morning", "Good afternoon", "Good evening", "Good night",
    "Please", "Thank you", "You are welcome", "Sorry", "Excuse me", "Goodbye", "See you", "Take care", "Nice to meet you",
  ]],
  ["Basic Needs", [
    "I need help", "I need water", "I need food", "I am hungry", "I am thirsty", "I need the bathroom",
    "I need medicine", "I need rest", "I am cold", "I am hot", "I have pain", "Please wait",
  ]],
  ["Food & Drink", [
    "Water", "Tea", "Coffee", "Milk", "Juice", "Breakfast", "Lunch", "Dinner", "Rice", "Bread",
    "Fruit", "Vegetables", "More", "Enough", "The bill please", "No spicy food",
  ]],
  ["Family & People", [
    "Mother", "Father", "Brother", "Sister", "Child", "Family", "Friend", "Partner", "Baby",
    "Doctor", "Nurse", "Teacher", "Manager", "Customer", "Neighbor",
  ]],
  ["Time & Calendar", [
    "Today", "Tomorrow", "Yesterday", "Now", "Later", "Soon", "Morning", "Afternoon", "Evening",
    "Night", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
  ]],
  ["Places & Travel", [
    "Home", "Work", "School", "Hospital", "Pharmacy", "Shop", "Market", "Restaurant", "Hotel",
    "Bus stop", "Train station", "Airport", "Bathroom", "Office", "Parking", "Go home",
  ]],
  ["Feelings & Emergency", [
    "I am happy", "I am sad", "I am tired", "I am worried", "I am scared", "I am angry",
    "I am okay", "I feel sick", "Emergency", "Call the police", "Call an ambulance", "I am lost",
    "I need a doctor", "Please stay with me",
  ]],
  ["Common Verbs & Questions", [
    "I want", "I need", "I like", "I do not like", "Go", "Come", "Wait", "Stop", "Start", "Help",
    "Open", "Close", "Call", "Send", "Show me", "What", "Where", "When", "Who", "Why", "How",
    "Do you understand", "I understand", "I do not understand", "Please repeat",
  ]],
  ["Daily Life", [
    "Phone", "Message", "Call me", "Money", "Ticket", "Key", "Door", "Book", "Bag", "Ready",
    "Finished", "Again", "Yes", "No", "Maybe", "One moment", "Come with me", "Thank you for waiting",
  ]],
  ["Accessibility", [
    "Please write it down", "Please speak slowly", "Please face me", "I am Deaf", "I am hard of hearing",
    "I use sign language", "Please use captions", "Please type here", "I need an interpreter", "Can you repeat that",
  ]],
];

export const DEFAULT_VOCABULARY: VocabularyItem[] = rows.flatMap(([category, phrases]) =>
  phrases.map((phrase, index) => ({
    id: "default-" + category.toLowerCase().replace(/[^a-z0-9]+/g, "-") + "-" + index,
    phrase,
    category,
  }))
);
