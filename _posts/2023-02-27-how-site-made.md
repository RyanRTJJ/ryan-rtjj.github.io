---
published: true
title: How I Made This Site
date: 2023-02-26 00:00:00 -500
categories: [miscellaneous]
tags: [websitebuilder,jekyll,ruby]
math: true
---
This site was made using Jekyll and was _very_ easy to make. Minimal code, super functional, beautiful, and modern. I started using the [Chirpy Starter repo](https://github.com/cotes2020/chirpy-starter/), following this awesome [YouTube walkthrough](https://www.youtube.com/watch?v=F8iOU1ci19Q) by creator Techno Tim. He gives pretty much all the instructions, but here are some idiosyncrasies that you might face along the way and some tips to help you get started.

# Dependencies
As this site was generated using Jekyll, which is written in Ruby, there are a bunch of installations to do if you don't typically use Ruby. In particular, make sure that the following are up-to-date (I use a Mac; so this advice is most useful for Mac users):
- Ruby
- Xcode (you might run into problems like `pkg-config` not being ablt to find compiler libraries like `openssl` otherwise)

# Out-of-the-box Functionalities
Right out of the box, this template supports all sorts of **mark-down** (including **emojis** &#x1F47E;, which is written in hexadecimal), **code blocks** (which I conveniently demonstrate by showing the front matter of my posts to enable **mathjax (latex)**):
```md
---
published: true
title: How Site Made
date: 2023-02-27 00:00:00 -500
categories: [hmm]
tags: [websitebuilder,jekyll]
math: true
---
```

$$\left( \frac{\text{omg }\textbf{it's a}}{\text{LateX}}\Rightarrow \text{ Block Equation} \right)$$

light / dark mode supported theme, **RSS**, and **search**! It's great; &#x1F49B; the Chirpy people (it's open-source).

# My Customizations
Because I wanted to change things like fonts and colors, I decided to self-host [`chirpy-static-assets`](https://github.com/cotes2020/chirpy-static-assets#readme). This entails following all the instructions on that page, with the difference of setting `env` to be EMPTY in `_config.yml`:
```yml
# Self-hosted static assets, optional › https://github.com/cotes2020/chirpy-static-assets
assets:
  self_host:
    enabled: true     # boolean, keep empty means false
    # specify the Jekyll environment, empty means both
    # only works if `assets.self_host.enabled` is 'true'
    env:  # [development|production]
``` 
This allows me to edit css files (I found the alternatives too troublesome). It was fairly straightforward, following the instructions on the github repo. For example, I found that editing fonts in `/Library/Ruby/Gems/2.6.0/jekyll-theme-chirpy-5.5.2/_sass/addon/commons.scss` allowed me to change fonts. Adding the font files was as easy as plopping them into `assets/lib/fonts/`. 

# Deployment:
## References
- [Firebase Hosting Tutorial #1 (and #2)](https://www.youtube.com/watch?v=mmmaeHBCTOw&list=PL4cUxeGkcC9he0kHAyiyr3dDO2xw0NWoP&index=1): this is great for getting started with Firebase accounts and CLI.
- [Getting started with Firebase Hosting (and GitHub Actions!)](https://www.youtube.com/watch?v=P0x0LmiknJc): this is great for using the Firebase CLI to get your site actually hosted.

## Steps I took to initialize Firebase in my repo:
### Get Firebase CLI (command-line-interface)
1. Create a Firebase project on [firebase.google.com](firebase.google.com).

2. Once done, in the control panel for the project, select "Hosting" to set up hosting.

3. Install Firebase tools by running `npm install -g firebase-tools`. This requires NodeJS, which, if you don't have (or have an outdated version), you may download / update by going to the [NodeJS website](https://nodejs.org/en/download/), downloading the appropriate file, and installing it.

### Initialize Firebase in your website directory
1. When we run `bundle exec jekyll s` (to spin up a locally hosted version of the website) or `JEKYLL_ENV=production bundle exec jekyll b` (to compile a folder of production files that you can upload to Firebase (or S3, or whatever)), jekyll actually uses the config files to generate the `_site` folder for us. The `_site` folder contains all the static code and assets that the website needs to render properly, which is all the information that our hosting service needs to have. But this automatic generation by jekyll also means that any Firebase files placed in `_site` are transient and will be overwritten upon every `bundle exec jekyll s` call. We hence just initialize firebase in our website's root folder by running `firebase login` and `firebase init hosting` there. Read the paragraph below (2) before you do this!

2. In the initialization wizard, Firebase will ask **'What do you want to use as your public directory?**, with the default being `public`. Because Jekyll puts everything in the `_site` folder, we want to use `_site` as the public directory instead. This initialization process also helpfully generates a `index.js` and `404.html` files for us, but Chirpy already has those, so don't override! Once done, you can test it by running `firebase deploy --only hosting`. It should output a temporary URL that looks something like: `https://firebase-project-name.web.app`, and it should render your website.

3. [Optional: emulators] If you wish to use firebase emulators (local versions of the deployed projects to see if they work), you may also (also in the root folder) run `firebase init emulators`, and run `firebase emulators:start` to spin it up.

4. [Optional: preview channels] If you wish to spin up a web (instead of local) version of a new version of the site (you made some big changes and want to see how it looks / send a public link to friends), you can deploy a preview channel by running `firebase hosting:channel:deploy some_preview_channel_name --expires 2d` (`2d` means expires in 2 days, for example). It will generate a new URL that will live for 2 days, looking something like: `https://firebase-project-name--preview-channel-name-RANDOM-STRING.web.app`.

5. [Optional: viewing / deleting channels] With potentially many channels, you may view all your channels by running `firebase hosting:channel:list`, and delete by running `firebase hosting:channel:delete some-preview-channel-name`.

### GitHub Actions
I did not spend time figuring this out, but because of Chirpy's use of submodules, I found that it may be easiest to just purge the Git out of this entire Chirpy repo. This means that your repo will no longer be a Git repo, which means that you can't `git pull` new updates that the Chirpy repo will receive from the open source community. If you were going to customize your own site and think that Chirpy as it is is a great starting point, this shouldn't  be a problem. Simply run `rm -rf .git` in the project root directory, and once more in `lib/assets` (the submodule).

# New Posts
Writing new posts entails having new `.md` content files in the `_posts` folder. Once you're ready to make your new post go live, just run `JEKYLL_ENV=production bundle exec jekyll b`, followed by `firebase deploy --only hosting`. You should see your new post go live almost immediately!