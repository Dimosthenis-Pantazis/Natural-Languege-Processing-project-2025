# Εργασία Ανάλυσης Φυσικής γλώσσας 2025 


## Εισαγωγή:
Ο λόγος που χρησιμοποιούμε σημασιολογική ανακατασκευή είναι διότι βοηθά στην βελτίωση της γραμματικής των κειμένων, στην διαμόρφωση των κειμένων όσο αφορά την έκφραση και την διατήρηση του αρχικού νοήματος. Το NLP είναι υπεύθυνο για την χρήση αυτομάτων παραφράσεων pipelines ( T5, Pegasus, BART)  και αξιολόγηση τους με βάση σκορ (Π.χ bleu, rouge, cosine). Επίσης, μέσω embeddings δημιουργεί δενδρόγραμμα και PCA/t-SNE για ευκολία κατανόησης των αποτελεσμάτων.

## Μεθοδολογία:
  Για το Α χρησιμοποιήθηκαν χειροποίητες γλωσσικές αντικαταστάσεις με βάση τη γραμματική.
  Για το Β χρησιμοποιήθηκαν seq2seq μοντέλα με στόχο τη δημιουργία πιο φυσικών εκδοχών προσφέροντας παραλλαγές στο ύφος και ροή παράλληλα  διατηρώντας το νόημα. 
  Για το C αξιοποιήθηκαν διάφορες συγκρίσεις κειμένου, ποιο συγκεκριμένα: 
    - BLEU: overlap σε n-grams.
    - ROUGE: recall/precision σε λέξεις/φράσεις.
    - Cosine Similarity με BERT: μετρά σημασιολογική εγγύτητα με βάση τα embeddings.
  Για να το επιτύχουμε μια τεχνική που εφαρμόσαμε είναι η συνάφεια συνημιτόνου
  με τύπο cos(x)= (A*B)/(|A| * |B|)

## Πειράματα & Αποτελέσματα:

Μπορείτε να βρείτε παραδείγματα πριν/μετά την ανακατασκευή στα παρακάτω αρχεία: 
1. A_viz_bert_pca.png
2. A_viz_bert_tsne.png
3. B_viz_bert_pca.png
4. B_viz_bert_tsne.png

Το τμήμα κώδικα το οποίο είναι αφιερωμένο στο παραδοτέο 2 είναι χωρισμένο σε 2 κομμάτια:

### 2.1 Custom Workflow (Δενδρόγραμμα)
- vocab, _ = zip(*Counter(preprocess(text1)).items()):
Καλεί την preprocess() και μέσω αυτής παίρνουμε μοναδικές λέξεις από το πρώτο κείμενο.

- preprocess(text): Μετατρεπει τις λεξεις σε tokens αποφεύγοντας στίξη και βγάζει ως έξοδο μια λίστα από lowercase λέξεις χωρίς σημεία στίξης.

- embeddings = embed_words(vocab): Καλεί την embed_words() στο tuple vocab
embed_words(vocab): Για κάθε λέξη w καλεί bert_model.encode(w) από την οποία επιστρέφεται ένα encoded vector.

- plot_dendrogram(embeddings, vocab, title):
X = np.vstack([embeddings[w] for w in vocab]) : Δημιουργεί πίνακα (n_words × dim).
Z = linkage (X, method="ward"): Υλοποίηση hierarchical clustering στον πίνακα Χ με μέθοδο Ward.
dendrogram (Z, labels=vocab, title): σχεδιάζει δένδρο με άκρα (leaves) τις λέξεις και κάθε κόμβος έχει ύψος = απόσταση μεταξύ clusters(Z).
Ύστερα με μεθόδους plt απεικονίζεταιι το δενδρόγραμμα που προκύπτει από τη συνάρτηση dendrogram()

### 2.2 Computational Analysis
clean_tokens(text): Παρόμοιο με preprocess (). Επιστρέφει λίστα λέξεων για κάθε πρόταση/κείμενο αλλά δεν εφαρμόζει stopword removal



- Εκπαίδευση Word2Vec/FastText και φόρτωση GloVe
1. all_texts = [text1, text2] + reconstructed['humarin'] + reconstructed['pegasus'] + reconstructed['bart'] + custom_outputs: Δημιουργείται λίστα με όλα τα κείμενα, ώστε να δημιουργηθεί ενιαίο corpus.
sentences = [clean_tokens(s) for t in all_texts for s in sent_tokenize(t)]: λίστα sentences από προτάσεις της οποίας κάθε στοιχείο είναι λίστα λέξεων.
sentences = [s for s in sentences if s]: Αφαιρεί προτάσεις που κατέληξαν άδειες.
w2v_model = ft_model = glove_model = None:  Αρχικοποιεί τις μεταβλητές στο None αν κάτι πάει στραβά.

2. w2v_model = Word2Vec (sentences, vector_size=100, window=5, min_count=1, epochs=20) και
 ft_model = FastText (sentences, vector_size=100, window=5, min_count=1, epochs=20):
Εκπαίδευση των τοπικών embeddings με Word2Vec και FastText που παρομοίως κάνουν κάθε λέξη ένα διάνυσμα 100 διαστάσεων, κοιτούν 5 λέξεις γύρω τους για context, κρατάει κάθε λέξη που εμφανίζεται έστω και 1 φορά και επαναλαμβάνεται 20 φορές για σταθερότητα. 
glove_model = api.load("glove-wiki-gigaword-50"): Φορτώνει έτοιμο pretrained μοντέλο GloVe.

- Cache BERT
bert_cache_sent[t] = bert_model.encode(t) : cache για πρόταση (sentence-level) BERT embeddings.
bert_cache_words[w] = bert_model.encode(w): cache για λέξη με BERT για αποφυγή επαναλαμβανόμενου υπολογισμού.

- Αξιοποίηση
methods = ["word2vec", "fasttext", "glove", "bert"]: Ορίζουμε ποιες μεθόδους θα εξετάσουμε.

get_embedding(): Παίρνει μια λέξη και μια μέθοδο και επιστρέφει το αντίστοιχο διάνυσμα.
1. Για Word2Vec/FastText: ψάχνει στο δικό τους λεξιλόγιο.
2. Για GloVe: κοιτάει στο key to index.
3. Για BERT: παίρνει τα embeddings που είχαμε ήδη αποθηκεύσει.
4. Αν η λέξη δεν υπάρχει στη μέθοδο επιστρέφει None.

- summary = {m: {"sent": [], "common": [], "align": [] }form in methods}:
Για κάθε μέθοδο κρατάμε τρεις λίστες με σκορ sent (ομοιότητα προτασης), common (ομοιότητα μονο στις κοινές λέξεις) και align (πόσο καλά αντιστοιχούν οι λέξεις της μίας πρότασης στις πιο κοντινές λέξεις της άλλης).

for tag, orig, rec in pairs:
    toks_o, toks_r = clean_tokens(orig), clean_tokens(rec)
    s_cos = cosine(bert_cache_sent[orig], bert_cache_sent[rec]):
Παίρνει κάθε ζευγάρι (original sentence, reconstructed sentence), καθαρίζει τις λέξεις και υπολογίζει τη συνολική ομοιότητα πρότασης (cosine similarity) με BERT embeddings.

for m in methods:
Για κάθε μέθοδο υπολογίζονται τρία πράγματα
1. sims_common = [1.0 for w in set(toks_o)&set(toks_r) if get_embedding(w,m) is not None]: Common words similarity δηλαδή αν μια λέξη υπάρχει και στην αρχική και στην ανακατασκευασμένη πρόταση, παίρνει score = 1
2. emb_o = [get_embedding(w,m) for w in toks_o if get_embedding(w,m) is not None]
    emb_r = [get_embedding(w,m) for w in toks_r if get_embedding(w,m) is not None]
Παίρνει όλα τα διανύσματα λέξεων για original (emb_o) και reconstructed (emb_r).
Και εξίσου υπολογίζει το alignment similarity μεσω του τμηματος κωδικα:
3. if emb_o and emb_r:

Υπολογίζει πίνακα αποστάσεων (cosine) για κάθε λέξη original vs κάθε λέξη reconstructed
    dists = cdist(np.vstack(emb_o), np.vstack(emb_r), metric="cosine"):
Υπολογίζει πίνακα αποστάσεων (cosine) για κάθε λέξη original vs κάθε λέξη reconstructed. Για κάθε λέξη original, βρίσκει την πιο κοντινή reconstructed λέξη.
sims_align = 1 - dists.min(axis=1):
Παίρνει μέσο όρο αυτών των ομοιοτήτων.
if s_cos: summary[m]["sent"].append(s_cos)
if sims_common: summary[m]["common"].append(np.mean(sims_common))
if sims_align is not None and hasattr(sims_align, '__len__') and len(sims_align) > 0:
    summary[m]["align"].append(np.mean(sims_align)):
Αν υπάρχει score, το προσθέτει στις λίστες. Ύστερα εκτυπώνει τους μέσους όρους για κάθε μέθοδο.

- PCA/t-SNE Visualization
Κύρια μέθοδος visualize_embeddings(method, items, prefix):
words = [w for _, b, a in items for w in clean_tokens(b)+clean_tokens(a)]:
Παίρνει όλες τις λέξεις από τα κείμενα (πριν και μετά την ανακατασκευή).
top_words = [w for w, _ in Counter(words).most_common(40)]:
Κρατάει τις 40 πιο συχνές λέξεις.
for label, before, after in items:
    for tag,text in [("before", before), ("after", after)]:
Για κάθε ζευγάρι
for w in clean_tokens(text):
if w in top_words:
Παίρνει τις λέξεις και κρατάει μόνο όσες είναι στις top 40.
e = get_embedding(w, method)
                if e is not None:
                    emb_list.append(e); labels.append(w); groups.append(f"{label}_{tag}")
Βρίσκει το embedding τους και αποθηκευει το διανυσμα, το «όνομα» και την ομαδα του (κειμενο1, κειμενο2, τροποποιημενο1, τροποποιημενο2)
PCA προβολή
X = np.vstack(emb_list): Ενώνει όλα τα embeddings σε έναν πίνακα.
Xp = PCA(n_components=2).fit_transform(X): Κάνει PCA για να τα κατεβάσει σε 2 διαστάσεις.
for g in set(groups):
    idxs=[i for i,gg in enumerate(groups) if gg==g]
    plt.scatter(Xp[idxs,0], Xp[idxs,1], label=g)
Κάθε ομάδα g παίρνει διαφορετικό χρώμα.
for i, lab in enumerate(labels): plt.text(Xp[i,0],Xp[i,1], lab,fontsize=8): Βάζει και τα ονόματα των λέξεων δίπλα στα σημεία.
plt.title(f"PCA - {method}"); plt.legend(); plt.savefig(f"{prefix}_{method}_pca.png"); plt.close()
Αποθηκεύει την εικόνα σε αρχείο .png με ανάλογο όνομα για την ομάδα.
t-SNE προβολή

- Xt=TSNE(n_components=2,init="pca",random_state=42,learning_rate="auto").fit_transform(X)
Παρόμοια λογική με PCA, αλλά η t-SNE είναι πιο «μη γραμμική» που σημαίνει οτι προσπαθεί να βρει φυσικά clusters, δηλαδή να ομαδοποιήσει λέξεις που έχουν παρόμοιο νόημα.
Κάνει πάλι scatter plot και αποθηκεύει σε άλλο .png

Χρήση των συναρτήσεων προβολής PCA και t-SNE
items_A=[("A",sent1,custom_outputs[0]),("A",sent2,custom_outputs[1])]:
Ζευγάρια προτάσεων τύπου A, δηλαδη τα custom παραδείγματα.
items_B=[(f"B_{m}",o,r) for m,outs in reconstructed.items() for o,r in zip(texts,outs)]:
Ζευγάρια από τα μοντέλα Humarin, Pegasus, Bart.
for m in methods:
    visualize_embeddings(m, items_A, "A_viz")
    visualize_embeddings(m, items_B, "B_viz")
Για κάθε μέθοδο φτιάχνει εικόνες μέσω τη visualize_embeddings().

## Συζήτηση:
Με βάση τα αποτελέσματα του cosine, παρατηρούμε ότι αποτυπώθηκαν με μεγάλη ευστοχία   το νόημα των κειμένων, όμως το Word2Vec / GloVe παρουσίασε δυσκολίες λόγου του περιορισμένου pool δεδομένων. 
 Μερικές προκλήσεις κατά την ανακατασκευή  ήταν τα συντακτικά λάθη και οι παραφράσεις που άλλαζαν το νόημα. 
 Η αυτοματοποίηση της διαδικασίας μέσω NLP θα απαιτούσε το συνδυασμό πολλαπλών σταδίων σε ένα ενιαίο pipeline. Ακόμα, το pipeline  θα έπρεπε να είναι ικανό να περιέχει ενσωματωμένο μέτρα αξιολόγησης όπως bleu και rouge και άλλες τεχνικές όπως PCA, t-SNE για την απεικόνιση τους.
 Υπήρξε ένας μεγάλος αριθμός διαφορών στην ποιότητα ανακατασκευής μεταξύ τεχνικών και βιβλιοθηκών.
χειροκίνητη ανακατασκευή: Οι χειροποίητοι κανόνες στόχευαν συγκεκριμένα σημεία των προτάσεων και παρείχαν αρκετά περιορισμένοι κάλυψη.

  - παραφραστικά μοντέλα: Τα μοντέλα δημιουργούσαν περισσότερο φυσικές ανακατασκευές αλλά το κάθε μοντέλο είχε διαφορετική προσέγγιση. Το humarin προτιμούσε ποιο "ελεύθερες" διατυπώσεις, το Pegasus εξειδικευόταν σε μικρές προτάσεις σε αντίθεση με το BART που παρήγαγε μεγαλύτερες σε μέγεθος παραλλαγές.
  - Ενσωματωμένες λέξεις:  Τα Word2Vec και FastText αντιμετώπιζαν με ευκολία λέξεις  προς λέξεις όμως έχαναν το συνολικό νόημα, σε αντίθεση με τα BERT embeddings που παρουσίαζαν σημασιολογική συνάφεια σε επίπεδο πρότασης.

## Συμπέρασμα:
Συμπεραίνοντας,  η σημασιολογική ανακατασκευή είναι μια περίπλοκη διαδικασία με κάθε προσέγγιση να έχουν πλεονεκτήματα και μειονεκτήματα. Οι κανόνες είναι αποτελεσματικοί για στοχευμένα τμήματα και λάθη αλλά είναι ανεπαρκείς για πολύπλοκα κείμενα. Τα μοντέλα NLP παρήγαγα φυσικά κείμενα και τα context-free μοντέλα (Word2Vec, FastText, GloVe) περιορίζονται σe τοπικό επίπεδο, ενώ τα context-sensitive embeddings (BERT) απέδωσαν καλύτερα την ολική σημασιολογία και τη συνάφεια προτάσεων. Βασική πρόκληση ήταν διατήρηση του αρχικού νοήματος των κειμένων και η διόρθωση συντακτικών λαθών.

## Βιβλιογραφία:
[Hugging face](https://huggingface.co/)

