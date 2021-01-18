Contributing to SAC
===================

**Thanks for taking the time to contribute to the SAC projects!**




Reporting bugs and asking questions
-----------------------------------

You can post questions or issues or feedback through the following channels:

* [GitHub Issues](https://github.com/petuum/autodist/issues): For bug reports and feature requests. 
* SAC slack space: TBD soon.

Procedures to contribute a feature:
----------------------

1. Break your work into small, single-purpose patches if possible. It's much
   harder to merge in a large change with a lot of disjoint features.
2. Submit the patch as a [GitHub pull request](https://github.com/petuum/autodist/pulls) against the master branch. Describe in as many details as possible what you are going to do in the pull request. Link it to [GitHub issues](https://github.com/petuum/autodist/issues) if there is any associated issue.
3. Make sure that your code passes the linter. 
4. Add new unit tests for your code.
5. Make sure that your code passes all (existing and added) unit tests.


PR Review Process
-----------------

For contributors who are in the SAC organization:

- When you first create a PR, add an reviewer to the `assignee` section.
- Assignees will review your PR and add `@author-action-required` label if further actions are required.
- Address their comments and remove `@author-action-required` label from the PR.
- Repeat this process until assignees approve your PR.
- Once the PR is approved, the author is in charge of ensuring the PR passes the build. Add `test-ok` label if the build succeeds.
- Committers will merge the PR once the build is passing.

For contributors who are not in the SAC organization:

- Your PRs will have assignees shortly. Assignees or PRs will be actively engaging with contributors to merge the PR.
- Please actively ping assignees after you address your comments!


Common Mistakes To Avoid
------------------------

-  **Did you add tests?** (Or if the change is hard to test, did you
   describe how you tested your change?)

   -  We have a few motivations for why we ask for tests:

      1. to help us tell if we break it later
      2. to help us tell if the patch is correct in the first place
         (yes, we did review it, but as Knuth says, “beware of the
         following code, for I have not run it, merely proven it
         correct”)

   -  When is it OK not to add a test? Sometimes a change can't be
      conveniently tested, or the change is so obviously correct (and
      unlikely to be broken) that it's OK not to test it. On the
      contrary, if a change is seems likely (or is known to be likely)
      to be accidentally broken, it's important to put in the time to
      work out a testing strategy.

-  **Is your PR too long?**

   -  It's easier for us to review and merge small PRs. Difficulty of
      reviewing a PR scales nonlinearly with its size.
   -  When is it OK to submit a large PR? It helps a lot if there was a
      corresponding design discussion in an issue, with sign off from
      the people who are going to review your diff. We can also help
      give advice about how to split up a large change into individually
      shippable parts. Similarly, it helps if there is a complete
      description of the contents of the PR: it's easier to review code
      if we know what's inside!

-  **Comments for subtle things?** In cases where behavior of your code
   is nuanced, please include extra comments and documentation to allow
   us to better understand the intention of your code.
-  **Did you add a hack?** Sometimes a hack is the right answer. But
   usually we will have to discuss it.
-  **Do you want to touch a very core component?** In order to prevent
   major regressions, pull requests that touch core components receive
   extra scrutiny. Make sure you've discussed your changes with the team
   before undertaking major changes.
-  **Want to add a new feature?** If you want to add new features,
   comment your intention on the related issue. Our team tries to
   comment on and provide feedback to the community. It's better to have
   an open discussion with the team and the rest of the community prior
   to building new features. This helps us stay aware of what you're
   working on and increases the chance that it'll be merged.
-  **Did you touch unrelated code to the PR?** To aid in code review,
   please only include files in your pull request that are directly
   related to your changes.




## Contributing to AutoDist: Detailed Steps

Refer to the following guidelines to contribute new functionality or bug fixes:

1. [Install](docs/usage/tutorials/installation.md) from source under development mode.
2. Following the above procesures to create pull requests
3. Use Prospector to lint the Python code: `prospector autodist`.
4. Add unit and/or integration tests for any new code you write.
5. Run unit and or integration tests in both CPU and GPU environments: `cd tests && python3 -m pytest -s --run-integration .`
where `--run-integration` is optional when only running unit tests.
