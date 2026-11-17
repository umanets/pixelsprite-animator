//! Is a sprite animatable, and as what? For the draft: from the caption's words, with the word lists and rules of
//! `pilot3d/dataset/select_inputs.py` (the 3D farm's input selection): a role word makes a person (even "cat
//! warrior"); otherwise bird / quadruped words make an animal unless the caption names an object made after an
//! animal ("eagle feather", "bear trap") or a body plan we cannot animate yet (fish, snake, monkey ...).
//! Russian / Ukrainian words are first mapped to their nearest English list word with the multilingual text encoder
//! (done by the worker, see `worker/classify.rs`), then the same rules apply.

use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Human,
    Quadruped,
    Bird,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Human, Kind::Quadruped, Kind::Bird];
    pub fn label(self) -> &'static str {
        match self {
            Kind::Human => "human",
            Kind::Quadruped => "quadruped",
            Kind::Bird => "bird",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "human" => Some(Kind::Human),
            "quadruped" => Some(Kind::Quadruped),
            "bird" => Some(Kind::Bird),
            _ => None,
        }
    }
}

pub const QUAD: &str = "dog puppy cat kitten kitty fox wolf bear panda horse pony foal deer elk moose reindeer antelope gazelle cow calf
bull ox yak bison buffalo pig piglet boar hog sheep lamb ram goat lion lioness tiger leopard cheetah panther lynx jaguar
cougar puma bobcat hyena jackal coyote elephant mammoth rhino rhinoceros hippo hippopotamus giraffe zebra camel llama
alpaca rabbit bunny hare mouse rat hamster gerbil squirrel chipmunk raccoon badger otter ferret weasel mink hedgehog
porcupine armadillo pangolin beaver marmot mole shrew vole lemming dormouse possum opossum wombat koala tapir quoll
tenrec dachshund terrier collie spaniel hound sheepdog doberman retriever poodle bulldog corgi husky beagle pug
chihuahua labrador mastiff dalmatian shiba akita greyhound whippet mau triceratops stegosaurus ankylosaurus
brachiosaurus diplodocus sabertooth smilodon thylacine unicorn donkey mule bobcat ocelot caracal serval meerkat
mongoose skunk capybara guinea";

pub const BIRD: &str = "bird owl owlet eagle hawk falcon kestrel parrot macaw cockatoo parakeet budgie duck duckling goose swan pigeon
dove crow raven magpie jay sparrow finch robin wren kingfisher woodpecker hummingbird toucan crane heron egret ibis
stork flamingo pelican seagull gull albatross vulture condor puffin snipe anhinga swallow swift lark canary cardinal
bluebird oriole starling bat bats songbird hornbill cuckoo nightingale blackbird";

pub const SKIP: &str = "penguin ostrich emu kiwi dodo chicken hen rooster chick turkey peacock kangaroo wallaby monkey ape gorilla chimp
chimpanzee lemur macaque tamarin marmoset gelada baboon orangutan fish snail butterfly frog toad lizard chameleon
crocodile alligator turtle tortoise crab beetle snake spider insect seal walrus sloth rex raptor velociraptor
pterodactyl pteranodon whale dolphin shark octopus squid jellyfish worm bee ant";

pub const ITEMS: &str = "bowl collar carrier cage toy yarn bed cushion stroller seat bottle mobile gem handle harness crate bib feeder
ribbon tag pad carvings gate leash pillow bassinet pacifier fountain playpen crib rocker scoop cot ring cloth chair
diaper kennel sweater raincoat rivet warmer washcloth clip dish bag bouncer condo basket swing disc tower stick cup
post seed toys food treat brush comb house hutch aquarium tank saddle blanket mat plate figurine statue plush
pacifier rattle lamp sign";

pub const NOT_ANIMAL: &str = "feather feathers nest mask trap totem antler antlers plush rubber figurine statue toy toys battering lily
mane egg eggs skull carving carved sculpture ornament weathervane kite balloon cookie cake pinata puppet costume tree
paradise lamp pillow mug cup plate sign banner flag tapestry rug pelt hide fur
pendant amulet brooch badge emblem patch sticker stamp coin keychain charm fang tooth teeth tusk claw horn
cheese milk meat sausage jerky statuette idol print shaped cutout";

pub const ROLE: &str = "man woman men women girl boy kid child person people warrior knight rider riding king queen prince princess
mage wizard witch sorcerer hunter archer soldier fighter ninja samurai pirate chef farmer merchant guard lord lady
folk elf dwarf orc goblin troll zombie skeleton vampire ghost monk priest cleric paladin bard thief rogue assassin
villager player athlete hero heroine captain sailor viking cowboy";

/// Common sprite nouns that are neither people nor animatable animals. They are only targets for mapping
/// Russian / Ukrainian words ("зелье" -> potion), so such words do not land on a random animal word instead.
pub const OTHER: &str = "sword axe bow arrow spear dagger hammer staff wand shield helmet armor armour boots gloves potion
bottle flask mushroom flower plant tree leaf rock stone chest box coin crown ring gem crystal diamond book scroll map
key apple bread cake pie fruit vegetable carrot fire flame ice water cloud lightning bomb skull bone robot dragon slime
monster demon angel castle house tower door ship boat car rocket planet star moon sun heart food drink cup hat cape
icon pixel art small big red blue green white black golden old young cute angry";

struct Lists {
    quad: HashSet<&'static str>,
    bird: HashSet<&'static str>,
    skip: HashSet<&'static str>,
    items: HashSet<&'static str>,
    not_animal: HashSet<&'static str>,
    role: HashSet<&'static str>,
}

fn lists() -> &'static Lists {
    static L: OnceLock<Lists> = OnceLock::new();
    L.get_or_init(|| {
        let set = |s: &'static str| s.split_whitespace().collect::<HashSet<_>>();
        Lists { quad: set(QUAD), bird: set(BIRD), skip: set(SKIP), items: set(ITEMS), not_animal: set(NOT_ANIMAL), role: set(ROLE) }
    })
}

/// Every English word the rules know (targets for mapping Russian / Ukrainian words).
pub fn vocabulary() -> Vec<&'static str> {
    let mut v: Vec<&'static str> =
        [QUAD, BIRD, SKIP, ITEMS, NOT_ANIMAL, ROLE, OTHER].iter().flat_map(|s| s.split_whitespace()).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Lower-case words of a caption (letters of any script).
pub fn words(caption: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in caption.chars() {
        if ch.is_alphabetic() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A caption word and the English word the rules saw for it (the same word for English captions).
#[derive(Debug, Clone)]
pub struct MappedWord {
    pub original: String,
    pub english: String,
    /// Cosine similarity of the mapping (1.0 = same word).
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct Classification {
    /// None = not animatable (for now).
    pub kind: Option<Kind>,
    pub reason: String,
}

/// The rules of select_inputs.py (`animal_kind` + the role-word override) on already-English words.
pub fn classify_mapped(words: &[MappedWord]) -> Classification {
    let l = lists();
    let show = |w: &MappedWord| {
        if w.original == w.english { format!("'{}'", w.original) } else { format!("'{}' ≈ '{}' ({:.2})", w.original, w.english, w.score) }
    };
    let find = |set: &HashSet<&str>| words.iter().find(|w| set.contains(w.english.as_str()));
    if let Some(w) = find(&l.role) {
        return Classification { kind: Some(Kind::Human), reason: format!("role word {}", show(w)) };
    }
    let animal = find(&l.bird).or_else(|| find(&l.quad));
    if let Some(w) = find(&l.not_animal) {
        return Classification { kind: None, reason: format!("an object, not a creature: {}", show(w)) };
    }
    if animal.is_none() {
        if let Some(w) = find(&l.items) {
            return Classification { kind: None, reason: format!("an item: {}", show(w)) };
        }
    }
    if let Some(w) = find(&l.skip) {
        return Classification { kind: None, reason: format!("body plan not supported yet: {}", show(w)) };
    }
    if let Some(w) = find(&l.bird) {
        return Classification { kind: Some(Kind::Bird), reason: format!("flying animal {}", show(w)) };
    }
    if let Some(w) = find(&l.quad) {
        return Classification { kind: Some(Kind::Quadruped), reason: format!("four-legged animal {}", show(w)) };
    }
    Classification { kind: None, reason: "no person or animal word in the caption".into() }
}

/// Russian / Ukrainian stems of common sprite subjects -> an English list word. A caption word matches a stem when
/// it starts with it and the rest is a short ending (inflection, diminutive). Checked before the embedding
/// fallback, which is noisy for single words.
const RU_UK: &[(&str, &str)] = &[
    // four-legged
    ("кошк", "cat"), ("кот", "cat"), ("котик", "cat"), ("котён", "kitten"), ("котен", "kitten"), ("кіт", "cat"), ("кішк", "cat"),
    ("кошен", "kitten"), ("собак", "dog"), ("пёс", "dog"), ("пес", "dog"), ("псин", "dog"), ("щен", "puppy"), ("цуцен", "puppy"),
    ("лошад", "horse"), ("конь", "horse"), ("коня", "horse"), ("кінь", "horse"), ("коник", "horse"), ("жереб", "foal"),
    ("пони", "pony"), ("волк", "wolf"), ("волч", "wolf"), ("вовк", "wolf"), ("медвед", "bear"), ("медвеж", "bear"),
    ("ведмед", "bear"), ("ведмеж", "bear"), ("ведмід", "bear"), ("лиса", "fox"), ("лисы", "fox"), ("лису", "fox"), ("лисиц", "fox"),
    ("лисён", "fox"), ("лисен", "fox"), ("лисичк", "fox"), ("олен", "deer"), ("лось", "moose"), ("лося", "moose"),
    ("заяц", "hare"), ("зайц", "hare"), ("заєць", "hare"), ("зайч", "hare"), ("кролик", "rabbit"), ("кроль", "rabbit"),
    ("мышь", "mouse"), ("мыши", "mouse"), ("мышк", "mouse"), ("миша", "mouse"), ("мишк", "mouse"), ("крыс", "rat"),
    ("щур", "rat"), ("ёж", "hedgehog"), ("ежик", "hedgehog"), ("ёжик", "hedgehog"), ("їжак", "hedgehog"), ("їжач", "hedgehog"),
    ("свин", "pig"), ("поросён", "piglet"), ("порос", "piglet"), ("кабан", "boar"), ("коров", "cow"), ("бык", "bull"),
    ("бик", "bull"), ("теля", "calf"), ("телён", "calf"), ("коза", "goat"), ("козёл", "goat"), ("козел", "goat"),
    ("цап", "goat"), ("овц", "sheep"), ("овеч", "sheep"), ("вівц", "sheep"), ("баран", "ram"), ("ягн", "lamb"),
    ("тигр", "tiger"), ("лев", "lion"), ("льв", "lion"), ("леопард", "leopard"), ("гепард", "cheetah"), ("пантер", "panther"),
    ("рысь", "lynx"), ("рись", "lynx"), ("ягуар", "jaguar"), ("гиен", "hyena"), ("гієн", "hyena"), ("шакал", "jackal"),
    ("слон", "elephant"), ("мамонт", "mammoth"), ("носорог", "rhino"), ("носоріг", "rhino"), ("бегемот", "hippo"),
    ("гегемот", "hippo"), ("жираф", "giraffe"), ("зебр", "zebra"), ("верблюд", "camel"), ("лам", "llama"),
    ("альпак", "alpaca"), ("белк", "squirrel"), ("білк", "squirrel"), ("бурундук", "chipmunk"), ("енот", "raccoon"),
    ("єнот", "raccoon"), ("барсук", "badger"), ("борсук", "badger"), ("выдр", "otter"), ("видр", "otter"),
    ("хорёк", "ferret"), ("тхір", "ferret"), ("ласк", "weasel"), ("хомяк", "hamster"), ("хом'як", "hamster"),
    ("хомʼяк", "hamster"), ("панд", "panda"), ("коал", "koala"), ("бобр", "beaver"), ("крот", "mole"), ("кріт", "mole"),
    ("опоссум", "opossum"), ("енот", "raccoon"), ("скунс", "skunk"), ("капибар", "capybara"), ("капібар", "capybara"),
    ("единорог", "unicorn"), ("єдиноріг", "unicorn"), ("осёл", "donkey"), ("осел", "donkey"), ("ослик", "donkey"),
    ("віслюк", "donkey"), ("мул", "mule"), ("трицератопс", "triceratops"), ("стегозавр", "stegosaurus"),
    ("такс", "dachshund"), ("терьер", "terrier"), ("тер'єр", "terrier"), ("хаски", "husky"), ("мопс", "pug"),
    ("корги", "corgi"), ("бульдог", "bulldog"), ("пудел", "poodle"), ("далматин", "dalmatian"), ("сурикат", "meerkat"),
    // flying
    ("птиц", "bird"), ("птич", "bird"), ("птах", "bird"), ("пташ", "bird"), ("сова", "owl"), ("сову", "owl"),
    ("совы", "owl"), ("совен", "owlet"), ("сови", "owl"), ("филин", "owl"), ("пугач", "owl"), ("сыч", "owl"),
    ("орёл", "eagle"), ("орел", "eagle"), ("орл", "eagle"), ("беркут", "eagle"), ("ястреб", "hawk"), ("яструб", "hawk"),
    ("сокол", "falcon"), ("сокіл", "falcon"), ("попуга", "parrot"), ("папуг", "parrot"),
    ("какаду", "cockatoo"), ("утк", "duck"), ("утён", "duckling"), ("утен", "duckling"), ("утят", "duckling"),
    ("качк", "duck"), ("качен", "duckling"), ("гус", "goose"), ("лебед", "swan"), ("лебід", "swan"), ("голуб", "pigeon"),
    ("ворон", "raven"), ("ворона", "crow"), ("галк", "crow"), ("сорок", "magpie"), ("воробей", "sparrow"),
    ("воробь", "sparrow"), ("горобец", "sparrow"), ("горобц", "sparrow"), ("синиц", "finch"), ("снегир", "finch"),
    ("снігур", "finch"), ("малиновк", "robin"), ("дятел", "woodpecker"), ("дятл", "woodpecker"), ("колибри", "hummingbird"),
    ("колібрі", "hummingbird"), ("тукан", "toucan"), ("журавл", "crane"), ("журавел", "crane"), ("цапл", "heron"),
    ("чапл", "heron"), ("аист", "stork"), ("лелек", "stork"), ("фламинго", "flamingo"), ("фламінго", "flamingo"),
    ("пеликан", "pelican"), ("пелікан", "pelican"), ("чайк", "gull"), ("альбатрос", "albatross"), ("стервятник", "vulture"),
    ("гриф", "vulture"), ("кондор", "condor"), ("ласточк", "swallow"), ("ластів", "swallow"), ("стриж", "swift"),
    ("жаворон", "lark"), ("жайворон", "lark"), ("канаре", "canary"), ("скворец", "starling"), ("шпак", "starling"),
    ("соловей", "nightingale"), ("солов", "nightingale"), ("кукушк", "cuckoo"), ("зозул", "cuckoo"), ("дрозд", "blackbird"),
    ("летуч", "bat"), ("кажан", "bat"), ("нетопыр", "bat"), ("нетопир", "bat"),
    // people / roles
    ("рыцар", "knight"), ("лицар", "knight"), ("воин", "warrior"), ("воїн", "warrior"), ("войн", "warrior"),
    ("солдат", "soldier"), ("боец", "fighter"), ("боєць", "fighter"), ("король", "king"), ("корол", "king"),
    ("королев", "queen"), ("королев", "queen"), ("принц", "prince"), ("принцес", "princess"), ("маг", "mage"),
    ("волшебн", "wizard"), ("чарівн", "wizard"), ("чародей", "sorcerer"), ("чаклун", "sorcerer"), ("колдун", "sorcerer"),
    ("ведьм", "witch"), ("відьм", "witch"), ("охотни", "hunter"), ("мисливц", "hunter"), ("мисливец", "hunter"),
    ("лучни", "archer"), ("лучниц", "archer"), ("стрілець", "archer"), ("стрелок", "archer"), ("ниндз", "ninja"),
    ("ніндз", "ninja"), ("самура", "samurai"), ("пират", "pirate"), ("пірат", "pirate"), ("повар", "chef"),
    ("кухар", "chef"), ("фермер", "farmer"), ("купец", "merchant"), ("купц", "merchant"), ("торговец", "merchant"),
    ("стражник", "guard"), ("страж", "guard"), ("вартов", "guard"), ("эльф", "elf"), ("ельф", "elf"), ("гном", "dwarf"),
    ("дворф", "dwarf"), ("орк", "orc"), ("гоблин", "goblin"), ("гоблін", "goblin"), ("тролл", "troll"), ("троль", "troll"),
    ("зомби", "zombie"), ("зомбі", "zombie"), ("скелет", "skeleton"), ("вампир", "vampire"), ("вампір", "vampire"),
    ("призрак", "ghost"), ("привид", "ghost"), ("монах", "monk"), ("чернец", "monk"), ("жрец", "priest"),
    ("священ", "priest"), ("паладин", "paladin"), ("бард", "bard"), ("вор", "thief"), ("воров", "thief"),
    ("злодій", "thief"), ("злодi", "thief"), ("разбойни", "rogue"), ("розбійни", "rogue"), ("ассасин", "assassin"),
    ("асасин", "assassin"), ("убийц", "assassin"), ("крестьян", "villager"), ("селян", "villager"), ("житель", "villager"),
    ("герой", "hero"), ("героїн", "heroine"), ("героин", "heroine"), ("капитан", "captain"), ("капітан", "captain"),
    ("моряк", "sailor"), ("матрос", "sailor"), ("викинг", "viking"), ("вікінг", "viking"), ("ковбо", "cowboy"),
    ("мужчин", "man"), ("чоловік", "man"), ("женщин", "woman"), ("жінк", "woman"), ("девушк", "girl"), ("дівчин", "girl"),
    ("девочк", "girl"), ("парень", "boy"), ("парн", "boy"), ("хлопец", "boy"), ("хлопц", "boy"), ("хлопчик", "boy"),
    ("мальчик", "boy"), ("ребён", "child"), ("ребен", "child"), ("дитин", "child"), ("человек", "person"),
    ("людин", "person"), ("люди", "people"), ("богатыр", "warrior"), ("богатир", "warrior"), ("казак", "warrior"),
    ("козак", "warrior"), ("спортсмен", "athlete"), ("игрок", "player"), ("гравець", "player"),
    // not animatable for now
    ("рыб", "fish"), ("риб", "fish"), ("змея", "snake"), ("змею", "snake"), ("змеи", "snake"), ("змія", "snake"),
    ("змій", "snake"), ("лягушк", "frog"), ("жаб", "frog"), ("черепах", "turtle"), ("краб", "crab"), ("паук", "spider"),
    ("павук", "spider"), ("насеком", "insect"), ("комах", "insect"), ("пингвин", "penguin"), ("пінгвін", "penguin"),
    ("куриц", "chicken"), ("курк", "hen"), ("петух", "rooster"), ("півен", "rooster"), ("півн", "rooster"),
    ("цыпл", "chick"), ("курча", "chick"), ("индюк", "turkey"), ("індик", "turkey"), ("павлин", "peacock"),
    ("павич", "peacock"), ("страус", "ostrich"), ("обезьян", "monkey"), ("мавп", "monkey"), ("горилл", "gorilla"),
    ("горил", "gorilla"), ("кенгуру", "kangaroo"), ("бабочк", "butterfly"), ("метелик", "butterfly"), ("улитк", "snail"),
    ("равлик", "snail"), ("крокодил", "crocodile"), ("ящер", "lizard"), ("ящір", "lizard"), ("акул", "shark"),
    ("кит", "whale"), ("дельфин", "dolphin"), ("дельфін", "dolphin"), ("осьминог", "octopus"), ("восьмини", "octopus"),
    ("пчел", "bee"), ("бджол", "bee"), ("муравь", "ant"), ("мураш", "ant"), ("жук", "beetle"), ("тюлен", "seal"),
    ("морж", "walrus"), ("ленивец", "sloth"), ("лінивец", "sloth"),
    ("перо", "feather"), ("перья", "feathers"), ("пір'я", "feathers"), ("пірʼя", "feathers"), ("гнезд", "nest"),
    ("гнізд", "nest"), ("маск", "mask"), ("капкан", "trap"), ("ловушк", "trap"), ("пастк", "trap"), ("череп", "skull"),
    ("клык", "fang"), ("ікл", "fang"), ("коготь", "claw"), ("кіготь", "claw"), ("шкур", "pelt"), ("игрушк", "toy"),
    ("іграшк", "toy"), ("статуэтк", "figurine"), ("статуетк", "figurine"), ("статуя", "statue"), ("яйц", "egg"),
    ("яєць", "eggs"), ("яйце", "egg"), ("рог", "horn"), ("ріг", "horn"), ("рога", "antlers"), ("роги", "antlers"),
    ("плюшев", "plush"), ("ошейник", "collar"), ("нашийник", "collar"), ("клетк", "cage"), ("клітк", "cage"),
    ("миска", "bowl"), ("мяч", "toy"), ("м'яч", "toy"),
];

/// Prefixes of words that start like a dictionary stem but mean something else (lamp, left, sand, magic ...).
const RU_UK_NOT: &[&str] = &["котел", "котл", "ламп", "левы", "левая", "левое", "левой", "лево", "песо", "песк", "песн",
    "магия", "магии", "магию", "магія", "магії", "китай", "ворот", "ворс", "свинец", "свинц"];

/// The English word for a Russian / Ukrainian caption word, from the built-in stems.
pub fn ru_uk_lookup(word: &str) -> Option<&'static str> {
    if word.is_ascii() {
        return None;
    }
    let w = word.replace('ё', "е");
    if RU_UK_NOT.iter().any(|x| w.starts_with(&x.replace('ё', "е"))) {
        return None;
    }
    let n = w.chars().count();
    let mut best: Option<(&str, usize)> = None;
    for (stem, en) in RU_UK {
        let s = stem.replace('ё', "е");
        if !w.starts_with(&s) {
            continue;
        }
        let sl = s.chars().count();
        let rest = n - sl;
        // short stems only take short endings ("кот" + "а", not "кот" + "лета")
        let max_rest = if sl <= 3 { 2 } else if sl <= 4 { 4 } else { 6 };
        if rest <= max_rest && best.is_none_or(|(_, l)| sl > l) {
            best = Some((en, sl));
        }
    }
    best.map(|(en, _)| en)
}

/// Classification without the text encoder: English words as they are, Russian / Ukrainian words through the
/// built-in stems.
pub fn classify_english(caption: &str) -> Classification {
    let w: Vec<MappedWord> = words(caption)
        .into_iter()
        .map(|w| match ru_uk_lookup(&w) {
            Some(en) => MappedWord { english: en.to_string(), original: w, score: 1.0 },
            None => MappedWord { english: w.clone(), original: w, score: 1.0 },
        })
        .collect();
    classify_mapped(&w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules() {
        assert_eq!(classify_english("pixel art icon of a knight").kind, Some(Kind::Human));
        assert_eq!(classify_english("cat warrior").kind, Some(Kind::Human));
        assert_eq!(classify_english("a white puppy").kind, Some(Kind::Quadruped));
        assert_eq!(classify_english("parrot").kind, Some(Kind::Bird));
        assert_eq!(classify_english("eagle feather").kind, None);
        assert_eq!(classify_english("goldfish").kind, None);
        assert_eq!(classify_english("penguin").kind, None);
        assert_eq!(classify_english("sword").kind, None);
    }
}
