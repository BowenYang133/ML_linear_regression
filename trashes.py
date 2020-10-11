#TF_IDF
vectorizer = TfidfVectorizer(min_df=1)
x_train_review_tfidf = vectorizer.fit_transform(x_train['review'])
x_test_review_tfidf = vectorizer.transform(x_test['review'])
#print(x_test_review_tfidf.shape)
#print(x_train_review_tfidf.shape)


clf = MultinomialNB()
clf.fit(x_train_review_bow, y_train)
y_pred = clf.predict(x_test_review_bow)
print("naive bow based test accuracy ;", accuracy_score(y_pred, y_test))

clf1 = MultinomialNB(alpha=1)
clf1.fit(x_train_review_tfidf,y_train)
y_pred_tfN = clf1.predict(x_test_review_tfidf)
print("naive tfidf test accuracy:", accuracy_score(y_test, y_pred_tfN))