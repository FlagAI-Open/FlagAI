# Contributing to BMTrain

We welcome everyone's effort to make the community and the package better. You are welcomed to propose an issue, make a pull request or help others in the community. All of the efforts are appreciated!

There are many ways that you can contribute to BMTrain:

- ✉️ Submitting an issue.
- ⌨️ Making a pull request.
- 🤝 Serving the community.

## Submitting an issue
You can submit an issue if you find bugs or require new features and enhancements. Here are some principles:

1. **Language.** It is better to write your issue in English so that more people can understand and help you more conveniently. 
2. **Search.** It is a good habit to search existing issues using the search bar of GitHub. Make sure there are no duplicated or similar issues with yours and if yes, check their solutions first.
3. **Format.** It is also very helpful to write the issue with a good writing style. We provide templates of common types of issues and everyone is encouraged to use these templates. If the templates do not fit in your issue, feel free to open a blank one.
4. **Writing style.** Write your issues in clear and concise words. It is also important to provide enough details for others to help. For example in a bug report, it is better to provide your running environment and minimal lines of code to reproduce it.

## Making a pull request (PR)
You can also write codes to contribute. The codes may include a bug fix, a new enhancement, or a new running example. Here we provide the steps to make a pull request:

1. **Combine the PR with an issue.** Make us and others know what you are going to work on. If your codes try to solve an existing issue, you should comment on the issue and make sure there are no others working on it. If you are proposing a new enhancement, submit an issue first and we can discuss it with you before you work on it.

2. **Fork the repository.** Fork the repository to your own GitHub space by clicking the "Fork" button. Then clone it on your disk and set the remote repo:
```git
$ git clone https://github.com/<your GitHub>/BMTrain.git
$ cd BMTrain
$ git remote add upstream https://github.com/OpenBMB/BMTrain.git
```

3. **Write your code.** Change to a new branch to work on your modifications. 
```git
$ git checkout -b your-branch-name
```
You are encouraged to think up a meaningful and descriptive name for your branch. 

4. **Make a pull request.** After you finish coding, you should first rebase your code and solve the conflicts with the remote codes:
```git
$ git fetch upstream
$ git rebase upstream/main
```
Then you can push your codes to your own repo:
```git
$ git push -u origin your-branch-name
```
Finally, you can make the pull request from your GitHub repo and merge it with ours. Your codes will be merged into the main repo after our code review.


## Serving the community

Besides submitting issues and PRs, you can also join our community and help others. Efforts like writing the documents, answering questions as well as discussing new features are appreciated and welcomed. It will also be helpful if you can post your opinions and feelings about using our package on social media.

We are now developing a reward system and all your contributions will be recorded and rewarded in the future.


