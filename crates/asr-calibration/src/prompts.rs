// SPDX-License-Identifier: GPL-3.0-or-later
//! What the user is asked to read. All text is written for this project (GPL with the code).
//!
//! Five kinds, mixed in every session (see [`session_mix`]):
//! * **Words**: single everyday words (needs, days, numbers, family).
//! * **Phrases**: 2-3 word phrases, plus minimal pairs (*sip, ship*) that differ only in a
//!   tongue sound, the sounds glossectomy affects most (t d n l s z sh ch j k g r th y).
//! * **Sentences**: everyday sentences, numbers/places, tongue-heavy and longer sentences.
//! * **Tongue twisters**: long sentences built from traditional rhymes; hard on purpose.
//! * **Story**: original short stories read a few sentences at a time, in order, continued
//!   across sessions: long, natural, continuous speech.
//!
//! Within a kind, sessions take the items recorded least so far, in list order.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    Word,
    Phrase,
    Sentence,
    TongueTwister,
    Story,
}

impl Kind {
    pub const ALL: [Kind; 5] = [
        Kind::Word,
        Kind::Phrase,
        Kind::Sentence,
        Kind::TongueTwister,
        Kind::Story,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Word => "Single word",
            Kind::Phrase => "Short phrase",
            Kind::Sentence => "Sentence",
            Kind::TongueTwister => "Tongue twister",
            Kind::Story => "Story",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub kind: Kind,
    pub text: String,
    /// e.g. "The Lighthouse Cat, part 3 of 8"
    pub context: Option<String>,
}

pub const WORDS: &[&str] = &[
    "water",
    "help",
    "pain",
    "doctor",
    "nurse",
    "toilet",
    "medicine",
    "breakfast",
    "lunch",
    "dinner",
    "tea",
    "coffee",
    "juice",
    "milk",
    "sugar",
    "salt",
    "yes",
    "no",
    "okay",
    "please",
    "sorry",
    "hello",
    "goodbye",
    "stop",
    "wait",
    "again",
    "slowly",
    "louder",
    "cold",
    "hot",
    "tired",
    "hungry",
    "thirsty",
    "sleep",
    "outside",
    "inside",
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "evening",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "twenty",
    "hundred",
    "telephone",
    "television",
    "kitchen",
    "bathroom",
    "garden",
    "daughter",
    "son",
    "sister",
    "brother",
    "mother",
    "father",
    "friend",
];

pub const PHRASES: &[&str] = &[
    "Thank you.",
    "Good morning.",
    "Good night.",
    "I'm fine.",
    "Not now.",
    "Come here.",
    "Sit down.",
    "Call the nurse.",
    "More water.",
    "Too hot.",
    "Very good.",
    "See you soon.",
    "Help me please.",
    "I'm thirsty.",
    "Turn it off.",
    "Turn it up.",
    "Open the door.",
    "Close the window.",
    "Where is it?",
    "What time?",
    "How are you?",
    "Love you.",
    "Let's go.",
    "Wait a minute.",
    "Not yet.",
    "All done.",
    "Right here.",
    "Left side.",
    "Take it slowly.",
    "Lights off.",
    "Too loud.",
    "Bring it here.",
    "Just a little.",
    "That's enough.",
    "No thank you.",
    "tea, key",
    "sip, ship",
    "light, right",
    "thin, sin",
    "dot, got",
    "chair, share",
    "jam, yam",
    "nap, lap",
    "rice, lice",
    "toe, go",
    "sink, think",
    "tie, die",
    "zoo, sue",
    "watch, wash",
    "bus, buzz",
    "led, red",
    "dear, gear",
    "tin, kin",
    "sea, she",
    "late, rate",
];

/// Everyday sentences (the first 20 were the original calibration set).
pub const SENTENCES: &[&str] = &[
    "Please call my daughter today.",        // k l d t
    "I need a glass of cold water.",         // n g l s k d t
    "Thank you for this lovely lunch.",      // th k l n ch
    "The doctor said the test went well.",   // d t s n l
    "Can you get me the red cup?",           // k g t r d
    "The train leaves at ten thirty.",       // t r n l th
    "Did you lock the door last night?",     // d l k s t n
    "Six sticks lie next to the stone.",     // s ks st l n
    "Jack and Jill climbed the hill.",       // j k l d
    "I would like some tea with sugar.",     // l k t th sh g
    "What time is the next appointment?",    // t n ks
    "My throat feels sore this morning.",    // th r s z n
    "Let me know when the car is ready.",    // l n k r d
    "Could you turn the television down?",   // k d t r l zh n
    "She chose cheese and juice for lunch.", // sh ch z j
    "Yesterday I walked to the river.",      // y s t r
    "Please speak a little more slowly.",    // s p k l t
    "The children are playing outside.",     // ch l d r n s
    "Give the green bag to Gary.",           // g r n
    "Turn the lights on in the kitchen.",    // t r n l ts k ch
    // ---- everyday needs ----
    "Can I have some more water please?",
    "I am feeling a little tired today.",
    "Please help me sit up in the bed.",
    "Could you open the window for me?",
    "I would like to go for a short walk.",
    "The food is too hot to eat right now.",
    "Please bring my glasses from the table.",
    "I need to take my medicine at noon.",
    "Can you call the nurse for me?",
    "My mouth feels dry this afternoon.",
    "I am hungry, what is for dinner?",
    "Please turn off the fan, it is cold.",
    "Where did you put my phone charger?",
    "I want to watch the news tonight.",
    "Could you read this letter to me?",
    "Let us sit in the garden for a while.",
    "I did not sleep well last night.",
    "The pain is better than it was yesterday.",
    "Please speak to the doctor about my diet.",
    "I would like soft food for lunch today.",
    // ---- people & social ----
    "Good morning, how are you doing today?",
    "It is so nice to see you again.",
    "Tell the children I love them very much.",
    "My brother is coming to visit on Saturday.",
    "Thank you for taking such good care of me.",
    "Happy birthday, I hope you have a lovely day.",
    "Say hello to your mother for me.",
    "We should all have dinner together soon.",
    "I am sorry, could you repeat that slowly?",
    "Please wait, I am trying to say something.",
    "Did you find the keys you were looking for?",
    "Let me think about it and tell you later.",
    "That is a very good idea, let us do it.",
    "I do not understand, please explain again.",
    "Our neighbours have a new little dog.",
    // ---- numbers, time, places ----
    "The appointment is on Tuesday at three o'clock.",
    "It is about twenty minutes from here by car.",
    "My phone number starts with nine eight four.",
    "I was born in the year nineteen sixty two.",
    "The pharmacy closes at eight in the evening.",
    "We need two litres of milk and a dozen eggs.",
    "The bus stop is next to the post office.",
    "Meet me at the hospital entrance at ten.",
    "Today is the twelfth of the month.",
    "The total bill came to forty five rupees.",
    // ---- tongue-heavy sounds (t d n l s z sh ch j k g r th) ----
    "Sally sells sweet tea on the street.",
    "Lucky lazy lions lie down to rest.",
    "Nine nice nurses need new notebooks.",
    "Thirty thin threads tied together tightly.",
    "Cheerful children chase chickens in the yard.",
    "Get the grey goat out of the garden.",
    "Red roses grow right round the road.",
    "Kate keeps the keys in the kitchen cupboard.",
    "Zoe zipped her jacket in the cold wind.",
    "Dad drove down to the dock at dawn.",
    "Shelly shares fresh fish with her friends.",
    "Jane enjoys juice and jam with her toast.",
    "Ten tall trees stand by the lake.",
    "The lazy cat sleeps on the soft rug.",
    "Leave the lamp on in the living room.",
    // ---- longer sentences ----
    "When the rain stops, we can go to the market and buy fresh vegetables.",
    "I have been practising my speech every day with the therapist.",
    "Please write down the name of the new medicine so I remember it.",
    "The train was late, so we waited at the station for an hour.",
    "My favourite song reminds me of the summer we spent at the beach.",
    "Could you check whether the doctor has sent the test results?",
    "After lunch I like to rest for a little while and read the paper.",
    "We planted tomatoes, chillies and beans in the small garden.",
    "If you are going to the shop, please get some bread and butter.",
    "The weather is getting warmer and the days are getting longer.",
];

/// Long sentences built from traditional tongue twisters.
pub const TWISTERS: &[&str] = &[
    "She sells seashells by the seashore, and the shells she sells are surely seashells from the shore.",
    "Peter Piper picked a peck of pickled peppers, but where is the peck of pickled peppers Peter Piper picked?",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood all day in the woods?",
    "Betty Botter bought some butter, but she said the butter is bitter, so she bought a bit of better butter.",
    "Red lorry, yellow lorry, rolling slowly down the long and lonely road to the little town.",
    "Fred fed Ted bread, and Ted fed Fred bread, then they both had tea together in the garden.",
    "I saw Susie sitting in a shoe shine shop, where she sits she shines, and where she shines she sits.",
    "Round the rugged rock the ragged rascal ran, rushing right along the river to reach the road.",
    "Kitty caught the kitten in the kitchen, and the kitten quickly climbed the cupboard to escape.",
    "Larry sent the latter a letter later, but the letter Larry sent was lost along the lane.",
    "Six slippery snails slid slowly seaward, leaving silver trails across the sandy shore.",
    "The thirty three thieves thought that they thrilled the throne throughout Thursday.",
    "Lesser leather never weathered wetter weather better than the leather that was tanned in the sun.",
    "Nine nimble noblemen nibbled nuts near the northern gate while the night grew dark and cold.",
    "Chester the cheetah chewed a chunk of cheap cheddar cheese and chose to share it with the children.",
    "Lily ladled little Letty's lentil soup, then let it cool a little on the long wooden table.",
    "Daddy draws doors and drives down to the dock, where the dark green boats are tied together.",
    "Can you can a can as a canner can can a can, or do you need a bigger can to can it?",
];

pub struct Story {
    pub title: &'static str,
    pub parts: &'static [&'static str],
}

pub const STORIES: &[Story] = &[
    Story {
        title: "The Lighthouse Cat",
        parts: &[
            "On a small island far out at sea, there stood an old white lighthouse. A keeper named Thomas lived there with a grey cat called Pepper. Every evening, Thomas climbed the long stairs to light the great lamp.",
            "Pepper always followed him, one step behind, her tail held high like a little flag. At the top, she sat by the window and watched the dark water. Thomas liked to say that she was the real keeper of the light.",
            "One stormy night, the wind howled and the rain beat against the glass. The lamp flickered, then went out completely. Thomas lit a candle and searched for the spare bulb, but it was nowhere to be found.",
            "Out on the black waves, a small fishing boat was trying to find its way home. Without the light, the sailors could not see the dangerous rocks. Thomas felt his heart pounding as he searched every shelf and drawer.",
            "Suddenly Pepper jumped down and ran to an old wooden box under the stairs. She scratched at the lid and meowed loudly until Thomas came to look. Inside, wrapped in a cloth, was the spare bulb.",
            "Thomas carried it up the stairs as fast as his tired legs would go. His hands were shaking, but he fitted the bulb and switched on the lamp. A bright beam swept across the sea once more.",
            "The little boat turned towards the light and slowly found the safe channel. In the morning, the fishermen rowed over to the island with a basket of fresh fish. They said they had come to thank the keeper.",
            "Thomas laughed and pointed at the grey cat sleeping in the sun. You should thank her, he said, she found the bulb when I could not. From that day on, Pepper got the best piece of fish every Friday.",
        ],
    },
    Story {
        title: "Grandmother's Garden",
        parts: &[
            "When Maya was seven, she spent the summer with her grandmother in a village near the hills. Her grandmother had a garden full of tomatoes, beans, chillies and bright yellow flowers. Every morning they watered the plants before the sun grew too hot.",
            "Grandmother taught Maya the names of every plant and every bird that visited. She showed her how to press a seed gently into the soil with one finger. Be patient, she said, good things take time to grow.",
            "Maya planted a row of sunflower seeds next to the fence. Each day she ran outside to check them, but nothing happened. After a week, she sat on the step and told her grandmother that the seeds were broken.",
            "Grandmother smiled and handed her a cup of warm milk. Seeds are not broken, she said, they are busy under the ground, where we cannot see them. Come back tomorrow and look again.",
            "The next morning, Maya found tiny green leaves pushing up through the dark earth. She shouted so loudly that the chickens ran in every direction. Grandmother came out laughing, wiping her hands on her apron.",
            "By the end of the summer, the sunflowers were taller than Maya. Their big golden faces turned slowly to follow the sun across the sky. Bees and butterflies visited them all day long.",
            "On her last day, Maya collected the seeds from the biggest flower and put them in a paper bag. She wrote her grandmother's name on it in careful letters. She promised to plant them in her own garden at home.",
            "Many years later, Maya has a garden of her own, full of tomatoes and tall yellow flowers. When her children ask why she loves sunflowers so much, she tells them about a patient grandmother and a row of seeds that were never broken.",
        ],
    },
    Story {
        title: "The Night Train",
        parts: &[
            "The night train to the hills left the city at exactly ten o'clock. Ravi found his seat by the window and put his small suitcase on the rack. Across from him sat an old man reading a newspaper by the light of a tiny lamp.",
            "As the train rolled out of the station, the city lights slowly disappeared behind them. Soon there was only darkness outside, broken now and then by the glow of a sleeping village. The wheels clicked and clattered in a steady rhythm.",
            "The old man folded his newspaper and asked Ravi where he was going. Ravi said he was visiting his sister, who had just had a baby girl. The old man smiled and said that he was going to see his grandson for the very first time.",
            "They talked for hours about their families, their work and the places they had lived. The old man had been a teacher for forty years in a small school by the river. He still remembered the names of many of his students.",
            "Around midnight, a man came through the carriage selling hot tea and biscuits. They bought two cups and drank them slowly, watching the moon rise over the dark fields. Ravi thought it was the best tea he had tasted in years.",
            "Later, the train stopped at a quiet station in the middle of nowhere. A light rain was falling, and a dog was sleeping under the bench on the platform. After a few minutes, the guard blew his whistle and they moved on.",
            "When Ravi woke up, the sun was rising over green hills covered in tea gardens. The air coming through the window was cool and fresh. The old man was already awake, holding a small wrapped present for his grandson.",
            "At the last station, they shook hands and wished each other well. Ravi watched the old man walk slowly down the platform towards a young family waiting by the gate. Then he picked up his suitcase and went to meet his new niece.",
        ],
    },
];

/// Every prompt, grouped by kind, in selection order.
pub fn all_prompts() -> Vec<Prompt> {
    let simple = |kind, list: &[&str]| -> Vec<Prompt> {
        list.iter()
            .map(|t| Prompt {
                kind,
                text: t.to_string(),
                context: None,
            })
            .collect()
    };
    let mut v = simple(Kind::Word, WORDS);
    v.extend(simple(Kind::Phrase, PHRASES));
    v.extend(simple(Kind::Sentence, SENTENCES));
    v.extend(simple(Kind::TongueTwister, TWISTERS));
    for s in STORIES {
        for (i, part) in s.parts.iter().enumerate() {
            v.push(Prompt {
                kind: Kind::Story,
                text: part.to_string(),
                context: Some(format!("{}, part {} of {}", s.title, i + 1, s.parts.len())),
            });
        }
    }
    v
}

/// The kind of a prompt text, if it is one of ours (recordings imported from elsewhere
/// have none).
pub fn kind_of(text: &str) -> Option<Kind> {
    all_prompts()
        .into_iter()
        .find(|p| p.text == text)
        .map(|p| p.kind)
}

/// How many items of each kind a session of `n` items contains. Order = the order they
/// are asked in: short warm-up words first, the story last.
pub fn session_mix(n: usize) -> Vec<(Kind, usize)> {
    let n = n.max(1);
    let story = if n >= 20 {
        2
    } else if n >= 5 {
        1
    } else {
        0
    };
    let twister = if n >= 8 { (n / 10).max(1) } else { 0 };
    let rest = n - story - twister;
    let words = (rest * 35 + 50) / 100;
    let phrases = (rest * 25 + 50) / 100;
    let sentences = rest - words - phrases;
    vec![
        (Kind::Word, words),
        (Kind::Phrase, phrases),
        (Kind::Sentence, sentences),
        (Kind::TongueTwister, twister),
        (Kind::Story, story),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_adds_up_and_covers_all_kinds() {
        for n in 1..=40 {
            assert_eq!(
                session_mix(n).iter().map(|x| x.1).sum::<usize>(),
                n,
                "n={n}"
            );
        }
        let m = session_mix(12);
        assert!(m.iter().all(|(_, c)| *c > 0), "{m:?}");
    }

    #[test]
    fn prompts_are_unique_and_classified() {
        let all = all_prompts();
        let mut texts: Vec<_> = all.iter().map(|p| p.text.as_str()).collect();
        texts.sort();
        texts.dedup();
        assert_eq!(texts.len(), all.len(), "duplicate prompt text");
        assert_eq!(kind_of("sip, ship"), Some(Kind::Phrase));
        assert_eq!(kind_of("water"), Some(Kind::Word));
        assert!(all.iter().filter(|p| p.kind == Kind::Story).count() >= 24);
    }
}
