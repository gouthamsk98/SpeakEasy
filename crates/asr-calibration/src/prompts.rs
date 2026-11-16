// SPDX-License-Identifier: GPL-3.0-or-later
//! What the user is asked to read. All text is written for this project (GPL with the code).
//!
//! Seven kinds, mixed in every session (see [`session_mix`]):
//! * **Words**: single everyday words (needs, days, numbers, family, short one-syllable words).
//! * **Word lists**: rows of short words grouped by sound, read with a pause after each:
//!   many short words per take, with silence in between.
//! * **Phrases**: 2-3 word phrases, plus minimal pairs (*sip, ship*) that differ only in a
//!   tongue sound, the sounds glossectomy affects most (t d n l s z sh ch j k g r th y).
//! * **Sentences**: everyday sentences, numbers/places, tongue-heavy and longer sentences.
//! * **Tongue twisters**: long sentences built from traditional rhymes, plus short ones;
//!   hard on purpose.
//! * **Reading passages**: short paragraphs made mostly of short everyday words.
//! * **Story**: original short stories read a few sentences at a time, in order, continued
//!   across sessions: long, natural, continuous speech.
//!
//! Within a kind, sessions take the items recorded least so far, in list order.
//!
//! Some items also get a recording [`Condition`] (close to / far from the microphone, a long
//! silence first, a pause in the middle, slow, fast, quiet, loud, head turned away), so the
//! personal model also learns the user's voice the way it sounds in real life.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    Word,
    Phrase,
    Sentence,
    TongueTwister,
    Story,
    WordList,
    Paragraph,
}

impl Kind {
    /// In the order a session asks them: short warm-ups first, long reading last.
    pub const ALL: [Kind; 7] = [
        Kind::Word,
        Kind::WordList,
        Kind::Phrase,
        Kind::Sentence,
        Kind::TongueTwister,
        Kind::Paragraph,
        Kind::Story,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Word => "Single word",
            Kind::Phrase => "Short phrase",
            Kind::Sentence => "Sentence",
            Kind::TongueTwister => "Tongue twister",
            Kind::Story => "Story",
            Kind::WordList => "Word list",
            Kind::Paragraph => "Reading passage",
        }
    }

    /// Extra advice shown with every item of this kind.
    pub fn hint(self) -> Option<&'static str> {
        match self {
            Kind::WordList => Some("Pause briefly after each word."),
            Kind::TongueTwister => Some("Take it slowly; getting it exactly right doesn't matter."),
            Kind::Paragraph | Kind::Story => {
                Some("Read at your normal pace; pausing to breathe is fine.")
            }
            _ => None,
        }
    }
}

/// How to say an item. Real life is not a quiet room with the mouth 30 cm from the
/// microphone; recording the same voice in different ways makes the model robust to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Condition {
    #[default]
    Normal,
    Close,
    Far,
    PauseBefore,
    PauseInside,
    Slow,
    Fast,
    Soft,
    Loud,
    TurnAway,
}

impl Condition {
    /// Every condition except Normal, in rotation order.
    pub const SPECIAL: [Condition; 9] = [
        Condition::Far,
        Condition::PauseBefore,
        Condition::Close,
        Condition::Slow,
        Condition::PauseInside,
        Condition::Soft,
        Condition::TurnAway,
        Condition::Fast,
        Condition::Loud,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Condition::Normal => "Normal",
            Condition::Close => "Close to the microphone",
            Condition::Far => "Far from the microphone",
            Condition::PauseBefore => "Long silence first",
            Condition::PauseInside => "Pause in the middle",
            Condition::Slow => "Slowly",
            Condition::Fast => "Quickly",
            Condition::Soft => "Quietly",
            Condition::Loud => "Loudly",
            Condition::TurnAway => "Head turned away",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Condition::Normal => "normal",
            Condition::Close => "close",
            Condition::Far => "far",
            Condition::PauseBefore => "silence first",
            Condition::PauseInside => "pause inside",
            Condition::Slow => "slow",
            Condition::Fast => "fast",
            Condition::Soft => "quiet",
            Condition::Loud => "loud",
            Condition::TurnAway => "turned away",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Condition::Normal => "",
            Condition::Close => {
                "Lean in close: your mouth about a hand's width (10 cm) from the microphone."
            }
            Condition::Far => {
                "Sit back: about an arm's length (1 metre) from the microphone. Speak normally."
            }
            Condition::PauseBefore => {
                "Press Record, then stay quiet until the screen says NOW SPEAK."
            }
            Condition::PauseInside => {
                "Stop at the pause mark for a slow count of three, then carry on."
            }
            Condition::Slow => "Say it slowly and clearly, as if to someone hard of hearing.",
            Condition::Fast => "Say it a bit faster than usual, as if you are in a hurry.",
            Condition::Soft => "Say it quietly, as if someone nearby is asleep.",
            Condition::Loud => "Say it loudly, as if calling to someone in the next room.",
            Condition::TurnAway => "Turn your head to one side, away from the screen, and say it.",
        }
    }

    /// Seconds of silence the screen asks for before speaking.
    pub fn quiet_before_secs(self) -> f32 {
        if self == Condition::PauseBefore {
            4.0
        } else {
            0.0
        }
    }

    /// Whether this condition makes sense for `p`.
    pub fn fits(self, p: &Prompt) -> bool {
        match self {
            Condition::PauseInside => {
                p.kind != Kind::WordList && p.text.split_whitespace().count() >= 4
            }
            _ => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub kind: Kind,
    pub text: String,
    /// e.g. "The Lighthouse Cat, part 3 of 8"
    pub context: Option<String>,
    pub condition: Condition,
}

impl Prompt {
    /// For [`Condition::PauseInside`]: the text split where the pause goes (after a comma
    /// near the middle if there is one, else in the middle).
    pub fn pause_split(&self) -> (String, String) {
        let w: Vec<&str> = self.text.split_whitespace().collect();
        if w.len() < 2 {
            return (self.text.clone(), String::new());
        }
        let mid = w.len() / 2;
        let at = (1..w.len())
            .filter(|&i| w[i - 1].ends_with(',') || w[i - 1].ends_with(';'))
            .min_by_key(|&i| i.abs_diff(mid))
            .filter(|&i| i.abs_diff(mid) <= w.len() / 4)
            .unwrap_or(mid);
        (w[..at].join(" "), w[at..].join(" "))
    }
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
    // short one-syllable words
    "cat",
    "dog",
    "sun",
    "bed",
    "cup",
    "map",
    "pen",
    "box",
    "hat",
    "bus",
    "car",
    "key",
    "shoe",
    "sock",
    "bag",
    "book",
    "door",
    "chair",
    "fish",
    "egg",
    "rice",
    "bread",
    "soup",
    "fork",
    "spoon",
    "knife",
    "plate",
    "bowl",
    "sink",
    "soap",
    "towel",
    "comb",
    "rain",
    "snow",
    "wind",
    "tree",
    "leaf",
    "bird",
    "ship",
    "sea",
    "sand",
    "rock",
    "road",
    "shop",
    "bank",
    "park",
    "train",
    "light",
    "dark",
    "red",
    "blue",
    "green",
    "white",
    "black",
    "big",
    "small",
    "fast",
    "left",
    "right",
    "up",
    "down",
    "open",
    "close",
    "push",
    "pull",
    "sit",
    "stand",
    "walk",
    "run",
    "eat",
    "drink",
    "read",
    "write",
    "sing",
    "think",
    "laugh",
    "cry",
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
    "Can you can a can as a canner can can a can, or do you need a bigger can to can it?",    // short ones
    "Red leather, yellow leather.",
    "Toy boat, toy boat, toy boat.",
    "Six sick sheep sat by the shed.",
    "Fresh fried fish, fish fresh fried.",
    "A big black bug bit a big black bear.",
    "Greek grapes, Greek grapes, Greek grapes.",
    "Truly rural, truly rural.",
    "Silly Sally swiftly shooed seven silly sheep.",
    "Two tiny tigers take two taxis to town.",
    "Clean clams crammed in clean cans.",
    "Tim took tea to Tom at ten.",
    "Dan's dog dug deep down by the dock.",
    "Lucky lizards like lemon lollies.",
    "Chuck checks cheap cheese chips.",
    "Shy Shelly says she shall sew sheets.",
    "Nine nice night nurses nursing nicely.",
    "Kate can't catch Kim's kite.",
    "Thin sticks, thick bricks.",
    "Lilly's little lamb licked a lemon.",
    "Good blood, bad blood, good blood, bad blood.",
];

/// Rows of short words grouped by sound; read with a pause after each word.
pub const WORD_LISTS: &[&str] = &[
    "cat, cap, cot, cut, coat, cake, cow",
    "tap, top, tip, tub, ten, tin, toe",
    "dad, dog, dip, door, duck, dish, dot",
    "sun, sit, sock, sand, seal, soup, sky",
    "shoe, shop, ship, shell, sheep, shut, shy",
    "chin, chip, chop, chair, cheek, chalk, chew",
    "jam, jar, jet, job, jug, joke, June",
    "lip, lap, log, lid, leg, lamp, lake",
    "red, run, rug, rat, ring, rope, rose",
    "no, nut, net, nap, nose, nine, neck",
    "go, gap, gum, gate, goat, gold, game",
    "thin, thick, thumb, three, thorn, thief, thud",
    "this, that, these, those, then, them, there",
    "yes, yell, yard, yawn, year, yolk, young",
    "zip, zoo, zero, zone, buzz, fizz, maze",
    "bat, bed, bin, bus, bag, ball, bone",
    "pin, pen, pan, pot, pig, pie, pear",
    "fan, fin, fog, fox, feet, fire, fork",
    "van, vet, vest, vine, voice, view, vote",
    "win, wet, web, wig, wall, wave, wood",
    "mat, mop, mud, mix, moon, milk, mouse",
    "hat, hen, hill, hop, hug, home, hand",
    "kick, kite, king, kettle, kiss, keep, kid",
    "one, two, three, four, five, six, seven, eight, nine, ten",
    "eleven, twelve, fifteen, eighteen, thirty, forty, fifty, sixty",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
    "January, February, March, April, May, June",
    "July, August, September, October, November, December",
    "tea, key, sea, knee, bee, pea, tree",
    "light, right, white, kite, night, bite, sight",
];

/// Short reading passages made mostly of short everyday words: (title, text).
pub const PARAGRAPHS: &[(&str, &str)] = &[
    (
        "My Dog Sam",
        "I have a small brown dog. His name is Sam. Each day we walk to the park and back. Sam likes to run on the grass and chase the red ball. When it rains, he sits by the door and looks at me. Then I get his coat, and we go out in the wet. He is my best friend.",
    ),
    (
        "The Shop on the Hill",
        "On Monday I went to the shop on the hill. I got milk, bread, eggs and a bag of rice. The man at the till was kind and slow. He asked if I was well, and I said yes. It was cold, so I took the bus home and made a cup of hot tea.",
    ),
    (
        "Rain in the Night",
        "The rain came down all night. It beat on the roof and ran down the glass. In the dark I lay in bed and could not sleep. At six the rain stopped and the sun came up. The birds began to sing, the air was fresh, and the road shone like a long wet mirror.",
    ),
    (
        "A Day at the Sea",
        "We took the train to the sea. The sand was hot, and the sky was a deep blue. The kids dug a big hole and made a fort with a flag on top. Dad fell asleep in his chair. At five we ate fish and chips, then went for a last swim in the cool, calm sea.",
    ),
    (
        "The Old Clock",
        "In the hall there is an old clock. It was made by my great grandfather, more than a hundred years ago. It ticks all day and all night, and it chimes each hour. When I was a child, I used to count the chimes in bed. Now my own son counts them too.",
    ),
    (
        "Making Soup",
        "To make soup, chop an onion, two carrots and a stick of celery. Fry them in a pot with a little oil. Add a tin of tomatoes, some water, salt and pepper. Let it boil, then turn the heat down. After half an hour, the soup is thick, warm and ready to eat.",
    ),
    (
        "The Garden in Spring",
        "In spring the garden wakes up. Green shoots push through the soil, and the trees grow new leaves. Bees buzz from rose to rose. I like to sit on the bench with a cup of tea and watch the birds build their nests. Each week there is something new to see.",
    ),
    (
        "At the Doctor",
        "Last week I saw the doctor. She asked me how I felt and if I had any pain. She checked my chest, my throat and my blood. Then she said that I was doing well and should keep up my exercises. I was glad, and I walked home in the sun.",
    ),
    (
        "The Lost Key",
        "This morning I could not find my key. I looked in my coat, my bag and under the bed. I checked the kitchen and the car. At last I sat down to think. Then I saw it, hanging in the door, just where I had left it the night before.",
    ),
    (
        "My Town",
        "My town is not big, but it has all I need. There is a market on Friday, a park with a pond, and a small cafe that sells the best cake. The streets are quiet at night. Most people know each other, and they always stop to say hello.",
    ),
    (
        "Learning to Ride",
        "When I was six, my father taught me to ride a bike. He held the seat and ran behind me down the lane. I said, don't let go, and he said, I won't. But he did, and I rode all the way to the gate by myself. I still remember how proud I felt.",
    ),
    (
        "Winter Night",
        "Snow fell softly on the roofs and the fields. Inside, the fire was warm and bright. We sat close together and told jokes and old tales. The cat lay on the rug and purred. Outside, the world was white and still, and the moon shone on the frozen pond.",
    ),
    (
        "The Bus Ride",
        "Every Tuesday I take the number nine bus to the city. I sit near the front, by the window. We pass the school, the church and the big green park. The driver knows my name and always says good morning. The ride takes twenty minutes, and I like to watch the people.",
    ),
    (
        "A Letter",
        "Dear Tom, thank you for your kind letter. I am feeling much better now, and I can walk to the end of the road and back. The doctors say I am making good progress. Please come and see us soon. We will have lunch in the garden if the weather is fine. With love, Ann.",
    ),
    (
        "The Kite",
        "On a windy day, Jack and his dad took a kite to the top of the hill. The kite was red and yellow, with a long blue tail. Jack held the string tight as it rose up into the sky. It dipped and spun, then flew high above the trees. Jack laughed and did not want to go home.",
    ),
    (
        "Market Day",
        "On Saturday the market fills the square. You can buy fruit, fish, fresh bread, pots and pans, and old books. The sellers shout their prices and the air smells of hot food. I go early, before the crowds, and I always buy a bag of sweet oranges and a bunch of flowers.",
    ),
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
                condition: Condition::Normal,
            })
            .collect()
    };
    let mut v = simple(Kind::Word, WORDS);
    v.extend(simple(Kind::Phrase, PHRASES));
    v.extend(simple(Kind::Sentence, SENTENCES));
    v.extend(simple(Kind::TongueTwister, TWISTERS));
    v.extend(simple(Kind::WordList, WORD_LISTS));
    for (i, (title, text)) in PARAGRAPHS.iter().enumerate() {
        v.push(Prompt {
            kind: Kind::Paragraph,
            text: text.to_string(),
            context: Some(format!("{title} ({} of {})", i + 1, PARAGRAPHS.len())),
            condition: Condition::Normal,
        });
    }
    for s in STORIES {
        for (i, part) in s.parts.iter().enumerate() {
            v.push(Prompt {
                kind: Kind::Story,
                text: part.to_string(),
                context: Some(format!("{}, part {} of {}", s.title, i + 1, s.parts.len())),
                condition: Condition::Normal,
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

/// How many items of each kind a session of `n` items contains, in the order they are
/// asked: short warm-up words first, long reading last.
pub fn session_mix(n: usize) -> Vec<(Kind, usize)> {
    let n = n.max(1);
    let step = |levels: &[(usize, usize)]| {
        levels
            .iter()
            .filter(|(min_n, _)| n >= *min_n)
            .map(|x| x.1)
            .next_back()
            .unwrap_or(0)
    };
    let story = step(&[(5, 1), (30, 2)]);
    let paragraph = step(&[(10, 1), (30, 2), (45, 3)]);
    let word_list = step(&[(7, 1), (20, 2), (35, 3)]);
    let twister = if n >= 8 { (n / 8).max(1) } else { 0 };
    let rest = n - story - paragraph - word_list - twister;
    let words = (rest * 35 + 50) / 100;
    let phrases = (rest * 25 + 50) / 100;
    let sentences = rest - words - phrases;
    vec![
        (Kind::Word, words),
        (Kind::WordList, word_list),
        (Kind::Phrase, phrases),
        (Kind::Sentence, sentences),
        (Kind::TongueTwister, twister),
        (Kind::Paragraph, paragraph),
        (Kind::Story, story),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_adds_up_and_covers_all_kinds() {
        for n in 1..=60 {
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
        assert_eq!(
            kind_of("Red leather, yellow leather."),
            Some(Kind::TongueTwister)
        );
        assert!(all.iter().filter(|p| p.kind == Kind::Paragraph).count() >= 12);
        assert!(all.iter().filter(|p| p.kind == Kind::WordList).count() >= 20);
    }

    #[test]
    fn pause_split_prefers_a_comma_near_the_middle() {
        let p = |t: &str| Prompt {
            kind: Kind::Sentence,
            text: t.into(),
            context: None,
            condition: Condition::PauseInside,
        };
        let (a, b) =
            p("If you are going to the shop, please get some bread and butter.").pause_split();
        assert_eq!(a, "If you are going to the shop,");
        assert_eq!(b, "please get some bread and butter.");
        let (a, b) = p("Can I have some more water please?").pause_split();
        assert_eq!(format!("{a} {b}"), "Can I have some more water please?");
        assert!(!a.is_empty() && !b.is_empty());
        assert!(!Condition::PauseInside.fits(&p("Thank you.")));
    }
}
