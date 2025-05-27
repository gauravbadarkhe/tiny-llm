const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");

const v = 0.03;
const rawText = fs.readFileSync("data/shakespeare.txt", "utf-8");
const bigText = rawText.toLowerCase().replace(/\n/g, " ");
const text = bigText.slice(0, 100000); // increase for better results
console.log(`${text}Loaded ${text.length} characters`);

const chars = [...new Set(text)];
const vocabSize = chars.length;

const charToIdx = Object.fromEntries(chars.map((ch, i) => [ch, i]));
const idxToChar = Object.fromEntries(chars.map((ch, i) => [i, ch]));

const encoded = [...text].map((char) => charToIdx[char]);

const seqLength = 100;
const inputs = [];
const labels = [];

for (let i = 0; i < encoded.length - seqLength; i++) {
  const inputSeq = encoded.slice(i, i + seqLength);
  const label = encoded[i + seqLength];

  inputs.push(inputSeq);
  labels.push(label);
}

// Correct data type for classification
// Input tensor: int32 (for embeddings)
const xs = tf.tensor2d(inputs, [inputs.length, seqLength], "int32");

// Label tensor: float32 (even though it's class indices!)
const ys = tf.tensor1d(labels, "float32"); // ✅ this avoids the tf.floor error
const model = tf.sequential();

model.add(
  tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 64,
    inputLength: seqLength,
  }),
);

model.add(
  tf.layers.lstm({
    units: 128,
    returnSequences: true,
    recurrentDropout: 0.1,
    dropout: 0.1,
  }),
);

model.add(
  tf.layers.lstm({
    units: 128,
    dropout: 0.1,
  }),
);

model.add(
  tf.layers.dense({
    units: vocabSize,
    activation: "softmax",
  }),
);

model.summary();

model.compile({
  optimizer: tf.train.adam(0.001),
  loss: "sparseCategoricalCrossentropy",
  metrics: ["accuracy"],
});

async function train() {
  await model.fit(xs, ys, {
    epochs: 20,
    batchSize: 128,
    validationSplit: 0.1,
    callbacks: tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience: 3,
    }),
  });

  await model.save(`file://model-${v}`);
  console.log("✅ Model Saved");
}

function sampleWithTemperature(probabilities, temperature = 1.0) {
  const logits = tf.div(tf.log(probabilities), tf.scalar(temperature));
  const scaled = tf.softmax(logits);
  const floatScaled = scaled.asType("float32"); // ✅ convert to float32
  const idx = tf.multinomial(floatScaled, 1).dataSync()[0];
  return idx;
}

async function generateText(seed, length, v = "1", temperature = 0.8) {
  const model = await tf.loadLayersModel(`file://model-${v}/model.json`);

  let inputSeq = [...seed.toLowerCase()].map((c) => charToIdx[c] || 0);

  if (inputSeq.length < seqLength) {
    inputSeq = Array(seqLength - inputSeq.length)
      .fill(0)
      .concat(inputSeq);
  } else {
    inputSeq = inputSeq.slice(-seqLength);
  }

  let generatedText = seed;

  for (let i = 0; i < length; i++) {
    const inputTensor = tf.tensor2d([inputSeq], [1, seqLength]);
    const prediction = model.predict(inputTensor).squeeze();
    const nextCharIdx = sampleWithTemperature(prediction, temperature);

    const nextChar = idxToChar[nextCharIdx];
    generatedText += nextChar;

    process.stdout.write(nextChar);
    inputSeq.push(nextCharIdx);
    inputSeq.shift();
  }

  console.log("\n📜 Generated Text:\n", generatedText);
}

// train();

generateText(
  // "first citizen: before we proceed any further, hear me speak.  all: speak, speak.  first citizen: you are all resolved rather to die than to famish?  all: resolved. resolved.  first citizen: first, you know caius marcius is chief enemy to the people.  all: we know't, we know't.  first citizen: let us kill him, and we'll have corn at our own price. is't a verdict?  all: no more talking on't; let it be done: away, away!  second citizen: one word, good citizens.  first citizen: we are accounted poor citizens, the patricians good. what authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them let us revenge this with our pikes, ere we become rakes: for the gods know i speak this in hunger for bread, not in thirst for revenge.  second citizen: would you proceed especially against caius marcius?  all: against him first: he's a very dog to the commonalty.  second citizen: consider you what services he has done for his country?  first citizen: very well; and could be content to give him good report fort, but that he pays himself with being proud.  second citizen: nay, but speak not maliciously.  first citizen: i say unto you, what he hath done famously, he did it to that end: though soft-conscienced men can be content to say it was for his country he did it to please his mother and to be partly proud; which he is, even till the altitude of his virtue.  second citizen: what he cannot help in his nature, you account a vice in him. you must in no way say he is covetous.  first citizen: if i must not, i need not be barren of accusations; he hath faults, with surplus, to tire in repetition. what shouts are these? the other side o' the city is risen: why stay we prating here? to the capitol!  all: come, come.  first citizen: soft! who comes here?  second citizen: worthy menenius agrippa; one that hath always loved the people.  first citizen: he's one honest enough: would all the rest were so!  menenius: what work's, my countrymen, in hand? where go you with bats and clubs? the matter? speak, i pray you.  first citizen: our business is not unknown to the senate; they have had inkling this fortnight what we intend to do, which now we'll show 'em in deeds. they say poor suitors have strong breaths: they shall know we have strong arms too.  menenius: why, masters, my good friends, mine honest neighbours, will you undo yourselves?  first citizen: we cannot, sir, we are undone already.  menenius: i tell you, friends, most charitable care have the patricians of you. for your wants, your suffering in this dearth, you may as well strike at the heaven with your staves as lift them against the roman state, whose course will on the way it takes, cracking ten thousand curbs of more strong link asunder than can ever appear in your impediment. for the dearth, the gods, not the patricians, make it, and your knees to them, not arms, must help. alack, you are transported by calamity thither where more attends you, and you slander the helms o' the state, who care for you like fathers, when you curse them as enemies.  first citizen: care for us! true, indeed! they ne'er cared for us yet: suffer us to famish, and their store-houses crammed with grain; make edicts for usury, to support usurers; repeal daily any wholesome act established against the rich, and provide more piercing statutes daily, to chain up and restrain the poor. if the wars eat us not up, they will; and there's all the love they bear us.  menenius: either you must confess yourselves wondrous malicious, or be accused of folly. i shall tell you a pretty tale: it may be you have heard it; but, since it serves my purpose, i will venture to stale 't a little more.  first citizen: well, i'll hear it, sir: yet you must not think to fob off our disgrace with a tale: but, an 't please you, deliver.  menenius: there was a time when all the body's members rebell'd against the belly, thus accused it: that only like a gulf it did remain i' the midst o' the body, idle and unactive, still cupboarding the viand, never bearing like labour with the rest, where the other instruments did see and hear, devise, instruct, walk, feel, and, mutually participate, did minister unto the appetite and affection common of the whole body. the belly answer'd--  first citizen: well, sir, what answer made the belly?  menenius: sir, i shall tell you. with a kind of smile, which ne'er came from the lungs, but even thus-- for, look you, i may make the belly smile as well as speak--it tauntingly replied to the discontented members, the mutinous parts that envied his receipt; even so most fitly as you malign our senators for that they are not such as you.  first citizen: your belly's answer? what! the kingly-crowned head, the vigilant eye, the counsellor heart, the arm our soldier, our steed the leg, the tongue our trumpeter. with other muniments and petty helps in this our fabric, if that they--  menenius: what then? 'fore me, this fellow speaks! what then? what then?  first citizen: should by the cormorant belly be restrain'd, who is the sink o' the body,--  menenius: well, what then?  first citizen: the former agents, if they did complain, what could the belly answer?  menenius: i will tell you if you'll bestow a small--of what you have little-- patience awhile, you'll hear the belly's answer.  first citizen: ye're long about it.  menenius: note me this, good friend; your most grave belly was deliberate, not rash like his accusers, and thus answer'd: 'true is it, my incorporate friends,' quoth he, 'that i receive the general food at first, which you do live upon; and fit it is, because i am the store-house and the shop of the whole body: but, if you do remember, i send it through the rivers of your blood, even to the court, the heart, to the seat o' the brain; and, through the cranks and offices of man, the strongest nerves and small inferior veins from me receive that natural competency whereby they live: and though that all at once, you, my good friends,'--this says the belly, mark me,--  first citizen: ay, sir; well, well.  menenius: 'though all at once cannot see what i do deliver out to each, yet i can make my audit up, that all from me do back receive the flour of all, and leave me but the bran.' what say you to't?  first citizen: it was an answer: how apply you this?  menenius: the senators of rome are this good belly, and you the mutinous members; for examine their counsels and their cares, digest things rightly touching the weal o' the common, you shall find no public benefit which you receive but it proceeds or comes from them to you and no way from yourselves. what do you think, you, the great toe of this assembly?  first citizen: i the great toe! why the great toe?  menenius: for that, being one o' the lowest, basest, poorest, of this most wise rebellion, thou go'st foremost: thou rascal, that art worst in blood to run, lead'st first to win some vantage. but make you ready your stiff bats and clubs: rome and her rats are at the point of battle; the one side must have bale. hail, noble marcius!  marcius: thanks. what's the matter, you dissentious rogues, that, rubbing the poor itch of your opinion, make yourselves scabs?  first citizen: we have ever your good word.  marcius: he that will give good words to thee will flatter beneath abhorring. what would you have, you curs, that like nor peace nor war? the one affrights you, the other makes you proud. he that trusts to you, where he should find you lions, finds you hares; where foxes, geese: you are no surer, no, than is the coal of fire upon the ice, or hailstone in the sun. your virtue is to make him worthy whose offence subdues him and curse that justice did it. who deserves greatness deserves your hate; and your affections are a sick man's appetite, who desires most that which would increase his evil. he that depends upon your favours swims with fins of lead and hews down oaks with rushes. hang ye! trust ye? with every minute you do change a mind, and call him noble that was now your hate, him vile that was your garland. what's the matter, that in these several places of the city you cry against the noble senate, who, under the gods, keep you in awe, which else would feed on one another? what's their seeking?  menenius: for corn at their own rates; whereof, they say, the city is well stored.  marcius: hang 'em! they say! they'll sit by the fire, and presume to know what's done i' the capitol; who's like to rise, who thrives and who declines; side factions and give out conjectural marriages; making parties strong and feebling such as stand not in their liking below their cobbled shoes. they say there's grain enough! would the nobility lay aside their ruth, and let me use my sword, i'll make a quarry with thousands of these quarter'd slaves, as high as i could pick my lance.  menenius: nay, these are almost thoroughly persuaded; for though abundantly they lack discretion, yet are they passing cowardly. but, i beseech you, what says the other troop?  marcius: they are dissolved: hang 'em! they said they were an-hungry; sigh'd forth proverbs, that hunger broke stone walls, that dogs must eat, that meat was made for mouths, that the gods sent not corn for the rich men only: with these shreds they vented their complainings; which being answer'd, and a petition granted them, a strange one-- to break the heart of generosity, and make bold power look pale--they threw their caps as they would hang them on the horns o' the moon, shouting their emulation.  menenius: what is granted them?  marcius: five tribunes to defend their vulgar wisdoms, of their own choice: one's junius brutus, sicinius velutus, and i know not--'sdeath! the rabble should have first unroof'd the city, ere so prevail'd with me: it will in time win upon power and throw forth greater themes for insurrection's arguing.  menenius: this is strange.  marcius: go, get you home, you fragments!  messenger: where's caius marcius?  marcius: here: what's the matter?  messenger: the news is, sir, the volsces are in arms.  marcius: i am glad on 't: then we shall ha' means to vent our musty superfluity. see, our best elders.  first senator: marcius, 'tis true that you have lately told us; the volsces are in arms.  marcius: they have a leader, tullus aufidius, that will put you to 't. i sin in envying his nobility, and were i any thing but what i am, i would wish me only he.  cominius: you have fought together.  marcius: were half to half the world by the ears and he. upon my party, i'ld revolt to make only my wars with him: he is a lion that i am proud to hunt.  first senator: then, worthy marcius, attend upon cominius to these wars.  cominius: it is your former promise.  marcius: sir, it is; and i am constant. titus lartius, thou shalt see me once more strike at tullus' face. what, art thou stiff? stand'st out?  titus: no, caius marcius; i'll lean upon one crutch and fight with t'other, ere stay behind this business.  menenius: o, true-bred!  first senator: your company to the capitol; where, i know, our greatest friends attend us.  titus:  cominius: noble marcius!  first senator:  marcius: nay, let them follow: the volsces have much corn; take these rats thither to gnaw their garners. worshipful mutiners, your valour puts well forth: pray, follow.  sicinius: was ever man so proud as is this marcius?  brutus: he has no equal.  sicinius: when we were chosen tribunes for the people,--  brutus: mark'd you his lip and eyes?  sicinius: nay. but his taunts.  brutus: being moved, he will not spare to gird the gods.  sicinius: be-mock the modest moon.  brutus: the present wars devour him: he is grown too proud to be so valiant.  sicinius: such a nature, tickled with good success, disdains the shadow which he treads on at noon: but i do wonder his insolence can brook to be commanded under cominius.  brutus: fame, at the which he aims, in whom already he's well graced, can not better be held nor more attain'd than by a place below the first: for what miscarries shall be the general's fault, though he perform to the utmost of a man, and giddy censure will then cry out of marcius 'o if he had borne the business!'  sicinius: besides, if things go well, opinion that so sticks on marcius shall of his demerits rob cominius.  brutus: come: half all cominius' honours are to marcius. though marcius earned them not, and all his faults to marcius shall be honours, though indeed in aught he merit not.  sicinius: let's hence, and hear how the dispatch is made, and in what fashion, more than his singularity, he goes upon this present action.  brutus: lets along.  first senator: so, your opinion is, aufidius, that they of rome are entered in our counsels and know how we proceed.  aufidius: is it not yours? what ever have been thought on in this state, that could be brought to bodily act ere rome had circumvention? 'tis not four days gone since i heard thence; these are the words: i think i have the letter here; yes, here it is. 'they have press'd a power, but it is not known whether for east or west: the dearth is great; the people mutinous; and it is rumour'd, cominius, marcius your old enemy, who is of rome worse hated than of you, and titus lartius, a most valiant roman, these three lead on this preparation whither 'tis bent: most likely 'tis for you: consider of it.'  first senator: our army's in the field we never yet made doubt but rome was ready to answer us.  aufidius: nor did you think it folly to keep your great pretences veil'd till when they needs must show themselves; which in the hatching, it seem'd, appear'd to rome. by the discovery. we shall be shorten'd in our aim, which was to take in many towns ere almost rome should know we were afoot.  second senator: noble aufidius, take your commission; hie you to your bands: let us alone to guard corioli: if they set down before 's, for the remove bring your army; but, i think, you'll find they've not prepared for us.  aufidius: o, doubt not that; i speak from certainties. nay, more, some parcels of their power are forth already, and only hitherward. i leave your honours. if we and caius marcius chance to meet, 'tis sworn between us we shall ever strike till one can do no more.  all: the gods assist you!  aufidius: and keep your honours safe!  first senator: farewell.  second senator: farewell",
  "123",
  3000,
  v,
  0.5,
);
// generateText("To be, or", 300, 0.02, 0.5);
